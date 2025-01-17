"""
CV Project
"""
import cv2
from matplotlib import pyplot as plt
import numpy as np
import Preprocessing
import Disparity

# header of ply file
ply_header = '''ply
format ascii 1.0
element vertex %(vert_num)d
property float x
property float y
property float z
property uchar red
property uchar green
property uchar blue
end_header
'''

# DEBUG FUNCTION
def show_images(images, cols=2, titles=None):
    '''Display a list of images in a single figure with matplotlib.'''

    assert((titles is None)or (len(images) == len(titles)))

    n_images = len(images)

    if titles is None: titles = ['Image (%d)' % i for i in range(1, n_images + 1)]
    fig = plt.figure()
    for n, (image, title) in enumerate(zip(images, titles)):
        a = fig.add_subplot(cols, np.ceil(n_images/float(cols)), n + 1)
        if image.ndim == 2:
            plt.gray()
        plt.imshow(image)
        a.set_title(title)
    fig.set_size_inches(np.array(fig.get_size_inches()) * n_images)
    plt.show()

# DEBUG FUNCTION
def show_img(img1, img2=None, cmap='gray'):
    ''' display one o two images'''

    if img2 is None:
        plt.imshow(img1, cmap)
    else:
        _ = plt.subplot(121), plt.imshow(img1, cmap)
        _ = plt.subplot(122), plt.imshow(img2, cmap)

    plt.show()

def drawlines(img1, img2, lines, pts1, pts2):
    ''' img1 - image on which we draw the epilines for the points in img2
        lines - corresponding epilines '''
    r, c = img1.shape

    img1 = cv2.cvtColor(img1, cv2.COLOR_GRAY2BGR)
    img2 = cv2.cvtColor(img2, cv2.COLOR_GRAY2BGR)
    for r, pt1, pt2 in zip(lines, pts1, pts2):
        color = tuple(np.random.randint(0, 255, 3).tolist())
        x0, y0 = map(int, [0, -r[2]/r[1]])
        x1, y1 = map(int, [c, -(r[2]+r[0]*c)/r[1] ])
        img1 = cv2.line(img1, (x0, y0), (x1, y1), color, 1)
        # img1 = cv2.circle(img1, tuple(pt1), 5, color, -1)
        # img2 = cv2.circle(img2, tuple(pt2), 5, color, -1)
    return img1, img2

def read_calib_file(fn, height, width):
    ''' Read the calibration file 
        and return the projection matrix Q'''
    # open file
    file = open(fn, 'r')
    # read focal length
    focal_length = np.float32(file.readline()[6: 14: 1])
    # read the baseline
    baseline = np.float32(file.readlines()[2].partition('=')[2].rstrip('\n'))
    Q = np.float32([
                    [1,  0,  0, -0.5 * width],
                    [0, -1,  0, 0.5 * height],
                    [0,  0,  0, -focal_length],
                    [0,  0, -1 / baseline, 0]
                   ])
    return Q

def feature_matching(img_l, img_r, features=0, sift_sigma=1.6, edge=10, 
                     contrast=0.04, layers=3, threshold=0.6):

    ''' find the best features matching '''
    # create a SIFT object
    sift = cv2.SIFT_create(nfeatures=features, nOctaveLayers=layers,
                                      contrastThreshold=contrast, 
                                      edgeThreshold=edge, sigma=sift_sigma)

    # find the keypoints and descriptors with SIFT
    kp1, des1 = sift.detectAndCompute(img_l, None)
    kp2, des2 = sift.detectAndCompute(img_r, None)

    # create BFMatcher object (brute force matching)
    # we preferred it to the flann matcher because flann it's pseudo-random
    brute_force_match = cv2.BFMatcher()
    matches = brute_force_match.knnMatch(des1, des2, k=2)

    pts1 = []
    pts2 = []

    for _, (img_l_match, img_r_match) in enumerate(matches):

        if img_l_match.distance < (threshold * img_r_match.distance):
            pts2.append(kp2[img_l_match.trainIdx].pt)
            pts1.append(kp1[img_l_match.queryIdx].pt)

    pts1 = np.float32(pts1)
    pts2 = np.float32(pts2)

    print('log:\n\tmatched: {} - selected: {}'.format(len(matches), len(pts1)))

    return pts1, pts2

def epipolar_geometry(img_l, img_r, fundamental_mat, pts1, pts2):
    ''' find and draw epilines '''
    # Find epilines corresponding to points in right image (second image) and
    # drawing its lines on left image
    lines1 = cv2.computeCorrespondEpilines(pts2.reshape(-1, 1, 2), 2, fundamental_mat)
    lines1 = lines1.reshape(-1, 3)
    img_l_ep, _ = drawlines(img_l, img_r, lines1, pts1, pts2)

    # Find epilines corresponding to points in left image (first image) and
    # drawing its lines on right image
    lines2 = cv2.computeCorrespondEpilines(pts1.reshape(-1, 1, 2), 1, fundamental_mat)
    lines2 = lines2.reshape(-1, 3)
    img_r_ep, _ = drawlines(img_r, img_l, lines2, pts2, pts1)

    return img_l_ep, img_r_ep

def sgbm_disparity_compute(img_l, img_r, win_size, block_size, ratio, disp_max_diff, sp_range):

    window_size = 3

    left_matcher = cv2.StereoSGBM_create(minDisparity=16,
                                        numDisparities=96,
                                        blockSize=block_size,
                                        P1=8*3*window_size**2,
                                        P2=32*3*window_size**2,
                                        disp12MaxDiff=disp_max_diff,
                                        uniquenessRatio=ratio,
                                        speckleWindowSize=win_size,
                                        speckleRange=sp_range,
                                        preFilterCap=63,
                                        mode=cv2.STEREO_SGBM_MODE_SGBM_3WAY)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(img_l, img_r)  
    dispr = right_matcher.compute(img_r, img_l)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filtered_img = wls_filter.filter(displ, img_l, None, dispr)

    return filtered_img

def bm_disparity_compute(img_l, img_r, win_size, block_size, ratio, disp_max_diff, sp_range):

    window_size = 3

    left_matcher = cv2.StereoBM_create( numDisparities=96,
                                        blockSize=block_size)

    right_matcher = cv2.ximgproc.createRightMatcher(left_matcher)

    # FILTER Parameters
    lmbda = 80000
    sigma = 1.2

    wls_filter = cv2.ximgproc.createDisparityWLSFilter(matcher_left=left_matcher)
    wls_filter.setLambda(lmbda)
    wls_filter.setSigmaColor(sigma)

    displ = left_matcher.compute(img_l, img_r)  
    dispr = right_matcher.compute(img_r, img_l)
    displ = np.int16(displ)
    dispr = np.int16(dispr)
    filtered_img = wls_filter.filter(displ, img_l, None, dispr)

    return filtered_img

def write_ply(file_name, verts, colors):
    ''' function to write a ply file '''

    verts = verts.reshape(-1, 3)
    colors = colors.reshape(-1, 3)
    verts = np.hstack([verts, colors])

    verts = verts[~np.isnan(verts).any(axis=1)]
    verts = verts[~np.isinf(verts).any(axis=1)]

    with open(file_name, 'w') as file:
        file.write(ply_header % dict(vert_num=len(verts)))
        np.savetxt(file, verts, '%f %f %f %d %d %d')

def reconstruction_3d(img_l_path, img_r_path,
                      sift_n_features, sift_sigma, edge, contrast, layers, threshold,
                      win_size, block_size, ratio, disp_max_diff, sp_range, algo: str):

    ''' Reconstruct a 3D scene from stereo image pairs. '''

    # read images
    img_l = cv2.imread(img_l_path, 0)
    img_r = cv2.imread(img_r_path, 0)

    # get height and width of the images
    # assuming have the same size
    height, width = img_l.shape[:2]

    # equalize histograms
    img_l_hist, img_r_hist = Preprocessing.intensity_offset_and_histogram_equalization(img_l, img_r)

    # filter
    img_fil_l, img_fil_r = Preprocessing.filter_application(img_l_hist, img_r_hist)

    # find the features in images using SIFT, brute forcing match and knn
    pts1, pts2 = feature_matching(img_fil_l, img_fil_r,
                                  sift_n_features, sift_sigma, edge,
                                  contrast, layers, threshold)

    # check if there are enough points
    # to calculate the fundamental matrix
    # if yes proceed
    # otherwise return False
    if len(pts1) < 9:
        return False

    # find fundamental matrix using opencv built-in function
    fundamental_mat, mask = cv2.findFundamentalMat(pts1, pts2, cv2.FM_LMEDS)

    # We select only inlier points
    pts1 = pts1[mask.ravel() == 1]
    pts2 = pts2[mask.ravel() == 1]

    # get images with epipolars lines drawed
    img_l_ep, img_r_ep = epipolar_geometry(img_l, img_r, fundamental_mat, pts1, pts2)

    # calculate the homographic matrices
    _, h_1, h_2 = cv2.stereoRectifyUncalibrated(pts1, pts2, fundamental_mat, (width, height))

    # rectify the images
    img_l_rect = cv2.warpPerspective(img_l, h_1, (width, height))
    img_r_rect = cv2.warpPerspective(img_r, h_2, (width, height))

    # calculate disparity map and apply post-filtering operation
    if(algo == 'sgbm'):
        disparity = sgbm_disparity_compute(img_l_rect, img_r_rect, win_size, block_size,
                                        ratio, disp_max_diff, sp_range)
    elif(algo == 'bm'):
        disparity = bm_disparity_compute(img_l_rect, img_r_rect, win_size, block_size,
                                            ratio, disp_max_diff, sp_range)
    elif(algo == 'sad'):
        disparity = Disparity.generate_disparity_map(img_l_rect, img_r_rect, name="sadImage", block_size=block_size)

    else:
        print("Algo error")
        assert(0)    

    return img_l, img_r, img_l_ep, img_r_ep, img_l_rect, img_r_rect, disparity, img_l_hist, img_r_hist, img_fil_l, img_fil_r

def disp_to_ply(img, calib, disp, ply_path):

    height, width = disp.shape[:2]
    # get matrix from calibration file
    Q = read_calib_file(calib, height, width)
    points = cv2.reprojectImageTo3D(-disp, -Q)

    colors = cv2.imread(img, cv2.COLOR_RGB2BGR)
    colors = cv2.cvtColor(colors, cv2.COLOR_RGB2BGR)

    mask = disp > disp.min()
    out_points = points[mask]
    out_colors = colors[mask]

    write_ply(ply_path, out_points, out_colors)
