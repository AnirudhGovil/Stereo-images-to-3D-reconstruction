    # -*- coding: utf-8 -*-

# Form implementation generated from reading ui file '.\stereovision.ui'
#
# Created by: PyQt5 UI code generator 5.14.1
#
# WARNING! All changes made in this file will be lost!


from PyQt5 import QtCore, QtGui, QtWidgets


class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        MainWindow.setObjectName("MainWindow")
        MainWindow.resize(1171, 685)
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap(".\\."), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        MainWindow.setWindowIcon(icon)
        self.centralwidget = QtWidgets.QWidget(MainWindow)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Fixed, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.centralwidget.sizePolicy().hasHeightForWidth())
        self.centralwidget.setSizePolicy(sizePolicy)
        self.centralwidget.setObjectName("centralwidget")
        self.verticalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget.setGeometry(QtCore.QRect(10, 10, 303, 481))
        self.verticalLayoutWidget.setObjectName("verticalLayoutWidget")
        self.form = QtWidgets.QVBoxLayout(self.verticalLayoutWidget)
        self.form.setContentsMargins(0, 0, 0, 0)
        self.form.setSpacing(5)
        self.form.setObjectName("form")
        self.leftFileSelection = QtWidgets.QHBoxLayout()
        self.leftFileSelection.setObjectName("leftFileSelection")
        self.filenameLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.filenameLabel.setObjectName("filenameLabel")
        self.leftFileSelection.addWidget(self.filenameLabel)
        self.loadFileButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.loadFileButton.setObjectName("loadFileButton")
        self.leftFileSelection.addWidget(self.loadFileButton)
        self.form.addLayout(self.leftFileSelection)
        self.rightFileSelection = QtWidgets.QHBoxLayout()
        self.rightFileSelection.setObjectName("rightFileSelection")
        self.form.addLayout(self.rightFileSelection)
        self.label_3 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_3.setObjectName("label_3")
        self.form.addWidget(self.label_3)

        # algos
        self.sgbmButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.sgbmButton.setObjectName("sgbmButton")
        self.leftFileSelection.addWidget(self.sgbmButton)

        self.bmButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.bmButton.setObjectName("bmButton")
        self.leftFileSelection.addWidget(self.bmButton)

        self.sadButton = QtWidgets.QPushButton(self.verticalLayoutWidget)
        self.sadButton.setObjectName("sadButton")
        self.leftFileSelection.addWidget(self.sadButton)


        self.siftParamsList = QtWidgets.QFormLayout()
        self.siftParamsList.setObjectName("siftParamsList")
        self.label_2 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_2.setObjectName("label_2")
        self.siftParamsList.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.label_2)
        self.edgeThresholdEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.edgeThresholdEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.edgeThresholdEdit.setObjectName("edgeThresholdEdit")
        self.siftParamsList.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.edgeThresholdEdit)
        self.label_4 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_4.setObjectName("label_4")
        self.siftParamsList.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.label_4)
        self.sigmaSiftEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.sigmaSiftEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sigmaSiftEdit.setObjectName("sigmaSiftEdit")
        self.siftParamsList.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.sigmaSiftEdit)
        self.contrastThresholdEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.contrastThresholdEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.contrastThresholdEdit.setObjectName("contrastThresholdEdit")
        self.siftParamsList.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.contrastThresholdEdit)
        self.label_5 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_5.setObjectName("label_5")
        self.siftParamsList.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.label_5)
        self.label_6 = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label_6.setObjectName("label_6")
        self.siftParamsList.setWidget(5, QtWidgets.QFormLayout.LabelRole, self.label_6)
        self.octaveEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.octaveEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.octaveEdit.setObjectName("octaveEdit")
        self.siftParamsList.setWidget(5, QtWidgets.QFormLayout.FieldRole, self.octaveEdit)
        self.label = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.label.setObjectName("label")
        self.siftParamsList.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.label)
        self.siftFeaturesEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.siftFeaturesEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.siftFeaturesEdit.setObjectName("siftFeaturesEdit")
        self.siftParamsList.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.siftFeaturesEdit)
        self.form.addLayout(self.siftParamsList)
        self.horizontalLayout_4 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_4.setObjectName("horizontalLayout_4")
        self.FlannLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.FlannLabel.setObjectName("FlannLabel")
        self.horizontalLayout_4.addWidget(self.FlannLabel)
        self.matcherThresholdLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.matcherThresholdLabel.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.matcherThresholdLabel.setObjectName("matcherThresholdLabel")
        self.horizontalLayout_4.addWidget(self.matcherThresholdLabel)
        self.form.addLayout(self.horizontalLayout_4)
        self.matcherThreshold = QtWidgets.QSlider(self.verticalLayoutWidget)
        self.matcherThreshold.setMinimum(1)
        self.matcherThreshold.setMaximum(100)
        self.matcherThreshold.setProperty("value", 80)
        self.matcherThreshold.setOrientation(QtCore.Qt.Horizontal)
        self.matcherThreshold.setObjectName("matcherThreshold")
        self.form.addWidget(self.matcherThreshold)
        self.horizontalLayout = QtWidgets.QHBoxLayout()
        self.horizontalLayout.setObjectName("horizontalLayout")
        self.form.addLayout(self.horizontalLayout)
        self.paramLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.paramLabel.setObjectName("paramLabel")
        self.form.addWidget(self.paramLabel)
        self.sgbmParams = QtWidgets.QFormLayout()
        self.sgbmParams.setObjectName("sgbmParams")
        self.winSizeLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.winSizeLabel.setObjectName("winSizeLabel")
        self.sgbmParams.setWidget(0, QtWidgets.QFormLayout.LabelRole, self.winSizeLabel)
        self.winSizeEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        sizePolicy = QtWidgets.QSizePolicy(QtWidgets.QSizePolicy.Minimum, QtWidgets.QSizePolicy.Fixed)
        sizePolicy.setHorizontalStretch(0)
        sizePolicy.setVerticalStretch(0)
        sizePolicy.setHeightForWidth(self.winSizeEdit.sizePolicy().hasHeightForWidth())
        self.winSizeEdit.setSizePolicy(sizePolicy)
        self.winSizeEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.winSizeEdit.setObjectName("winSizeEdit")
        self.sgbmParams.setWidget(0, QtWidgets.QFormLayout.FieldRole, self.winSizeEdit)
        self.sgbmBlockSizeLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.sgbmBlockSizeLabel.setObjectName("sgbmBlockSizeLabel")
        self.sgbmParams.setWidget(1, QtWidgets.QFormLayout.LabelRole, self.sgbmBlockSizeLabel)
        self.sgbmBlockSizeEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.sgbmBlockSizeEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.sgbmBlockSizeEdit.setObjectName("sgbmBlockSizeEdit")
        self.sgbmParams.setWidget(1, QtWidgets.QFormLayout.FieldRole, self.sgbmBlockSizeEdit)
        self.ratioLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.ratioLabel.setObjectName("ratioLabel")
        self.sgbmParams.setWidget(2, QtWidgets.QFormLayout.LabelRole, self.ratioLabel)
        self.ratioEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.ratioEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.ratioEdit.setObjectName("ratioEdit")
        self.sgbmParams.setWidget(2, QtWidgets.QFormLayout.FieldRole, self.ratioEdit)
        self.dispMaxLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.dispMaxLabel.setObjectName("dispMaxLabel")
        self.sgbmParams.setWidget(3, QtWidgets.QFormLayout.LabelRole, self.dispMaxLabel)
        self.dsipMaxEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.dsipMaxEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.dsipMaxEdit.setObjectName("dsipMaxEdit")
        self.sgbmParams.setWidget(3, QtWidgets.QFormLayout.FieldRole, self.dsipMaxEdit)
        self.spakleRangeLabel = QtWidgets.QLabel(self.verticalLayoutWidget)
        self.spakleRangeLabel.setObjectName("spakleRangeLabel")
        self.sgbmParams.setWidget(4, QtWidgets.QFormLayout.LabelRole, self.spakleRangeLabel)
        self.spakleEdit = QtWidgets.QLineEdit(self.verticalLayoutWidget)
        self.spakleEdit.setAlignment(QtCore.Qt.AlignRight|QtCore.Qt.AlignTrailing|QtCore.Qt.AlignVCenter)
        self.spakleEdit.setObjectName("spakleEdit")
        self.sgbmParams.setWidget(4, QtWidgets.QFormLayout.FieldRole, self.spakleEdit)
        self.form.addLayout(self.sgbmParams)
        self.viewTypeCombo = QtWidgets.QComboBox(self.centralwidget)
        self.viewTypeCombo.setGeometry(QtCore.QRect(930, 20, 231, 22))
        self.viewTypeCombo.setObjectName("viewTypeCombo")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.viewTypeCombo.addItem("")
        self.horizontalLayoutWidget = QtWidgets.QWidget(self.centralwidget)
        self.horizontalLayoutWidget.setGeometry(QtCore.QRect(320, 60, 841, 611))
        self.horizontalLayoutWidget.setObjectName("horizontalLayoutWidget")
        self.imageView = QtWidgets.QHBoxLayout(self.horizontalLayoutWidget)
        self.imageView.setContentsMargins(0, 0, 0, 0)
        self.imageView.setObjectName("imageView")
        self.verticalLayoutWidget_2 = QtWidgets.QWidget(self.centralwidget)
        self.verticalLayoutWidget_2.setGeometry(QtCore.QRect(10, 510, 301, 80))
        self.verticalLayoutWidget_2.setObjectName("verticalLayoutWidget_2")
        self.verticalLayout = QtWidgets.QVBoxLayout(self.verticalLayoutWidget_2)
        self.verticalLayout.setContentsMargins(0, 0, 0, 0)
        self.verticalLayout.setObjectName("verticalLayout")
        self.horizontalLayout_3 = QtWidgets.QHBoxLayout()
        self.horizontalLayout_3.setObjectName("horizontalLayout_3")
        self.defaultButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.defaultButton.setObjectName("defaultButton")
        self.horizontalLayout_3.addWidget(self.defaultButton)
        self.okButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.okButton.setObjectName("okButton")
        self.horizontalLayout_3.addWidget(self.okButton)
        self.verticalLayout.addLayout(self.horizontalLayout_3)
        self.generatePLYButton = QtWidgets.QPushButton(self.verticalLayoutWidget_2)
        self.generatePLYButton.setObjectName("generatePLYButton")
        self.verticalLayout.addWidget(self.generatePLYButton)
        self.workInProgressLabel = QtWidgets.QLabel(self.centralwidget)
        self.workInProgressLabel.setGeometry(QtCore.QRect(10, 600, 55, 51))
        self.workInProgressLabel.setText("")
        self.workInProgressLabel.setObjectName("workInProgressLabel")
        MainWindow.setCentralWidget(self.centralwidget)

        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        _translate = QtCore.QCoreApplication.translate
        MainWindow.setWindowTitle(_translate("MainWindow", "3D reconstruction"))
        self.filenameLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">File</span></p></body></html>"))
        self.loadFileButton.setText(_translate("MainWindow", "Load Images"))
        self.label_3.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">SIFT Parameters:</span></p></body></html>"))
        self.label_2.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Edge threshold</span></p></body></html>"))

        self.sgbmButton.setText(_translate("MainWindow", "Use SGBM"))
        self.bmButton.setText(_translate("MainWindow", "Use BM"))
        self.sadButton.setText(_translate("MainWindow", "Use SAD"))

        self.edgeThresholdEdit.setText(_translate("MainWindow", "10"))
        self.label_4.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Sigma</span></p></body></html>"))
        self.sigmaSiftEdit.setToolTip(_translate("MainWindow", "<html><head/><body><p>Default None</p></body></html>"))
        self.sigmaSiftEdit.setWhatsThis(_translate("MainWindow", "<html><head/><body><p>Default None</p></body></html>"))
        self.sigmaSiftEdit.setText(_translate("MainWindow", "1.6"))
        self.contrastThresholdEdit.setText(_translate("MainWindow", "0.04"))
        self.label_5.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Contrast threshold</span></p></body></html>"))
        self.label_6.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">N° Octave layers</span></p></body></html>"))
        self.octaveEdit.setText(_translate("MainWindow", "3"))
        self.label.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">N° Features</span></p></body></html>"))
        self.siftFeaturesEdit.setText(_translate("MainWindow", "0"))
        self.FlannLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt; font-weight:600;\">Matcher threshold:</span></p></body></html>"))
        self.matcherThresholdLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">0.80</span></p></body></html>"))
        self.paramLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:11pt; font-weight:600;\">Parameters:</span></p></body></html>"))
        self.winSizeLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Speckle Window Size</span></p></body></html>"))
        self.winSizeEdit.setText(_translate("MainWindow", "400"))
        self.sgbmBlockSizeLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Block size</span></p></body></html>"))
        self.sgbmBlockSizeEdit.setText(_translate("MainWindow", "7"))
        self.ratioLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Uniqueness Ratio</span></p></body></html>"))
        self.ratioEdit.setText(_translate("MainWindow", "12"))
        self.dispMaxLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Disp Max Diff</span></p></body></html>"))
        self.dsipMaxEdit.setText(_translate("MainWindow", "1"))
        self.spakleRangeLabel.setText(_translate("MainWindow", "<html><head/><body><p><span style=\" font-size:10pt;\">Spakle range</span></p></body></html>"))
        self.spakleEdit.setText(_translate("MainWindow", "5"))
        self.viewTypeCombo.setItemText(0, _translate("MainWindow", "Normal"))
        self.viewTypeCombo.setItemText(1, _translate("MainWindow", "Histogram equalization"))
        self.viewTypeCombo.setItemText(2, _translate("MainWindow", "Filter image"))
        self.viewTypeCombo.setItemText(3, _translate("MainWindow", "Epiolar geometry"))
        self.viewTypeCombo.setItemText(4, _translate("MainWindow", "Rectified"))
        self.viewTypeCombo.setItemText(5, _translate("MainWindow", "Disparity"))
        self.viewTypeCombo.setItemText(6, _translate("MainWindow", "Equalized Disparity"))
        self.defaultButton.setText(_translate("MainWindow", "Default"))
        self.okButton.setText(_translate("MainWindow", "OK"))
        self.generatePLYButton.setText(_translate("MainWindow", "Generate PLY"))
