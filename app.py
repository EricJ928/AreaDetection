"""
The project is originally created on 01-Sep-2020 by Eric.

@author: Junxiang(Eric) JIA

***** Version 2.0 *****
The version 2.0 is updated on 21-Aug-2021. Compare to version 1.0, the update version 2.0 uses a different
approach (OpenCV rather than customized pattern recognation algorithm).
"""

import numpy as np

import sys
import matplotlib.pyplot as plt
from matplotlib.backends.backend_qt5agg import FigureCanvasQTAgg as FigureCanvas

from PyQt5 import QtCore, QtWidgets, QtGui
from PyQt5.QtWidgets import QFileDialog
from ui import Ui_MainWindow

import cal

import matplotlib
matplotlib.use('Qt5Agg')


class MplCanvas(FigureCanvas):
    def __init__(self, parent=None, width=5, height=5, dpi=100):
        fig = plt.figure(figsize=(width, height), dpi=dpi)
        plt.subplot(111)
        # fig = Figure(figsize=(width, height), dpi=dpi)
        # self.axes = fig.add_subplot(111)
        super(MplCanvas, self).__init__(fig)
        # fig.tight_layout()
        # FigureCanvas.updateGeometry(self)


class staticplot(MplCanvas):
    def __init__(self, *args, **kwargs):
        MplCanvas.__init__(self, *args, **kwargs)

    def plotImg(self, Map, len_x, len_y):
        CM_mean = Map[np.nonzero(Map)].mean()
        CM_std = Map[np.nonzero(Map)].std()

        # vm = [CM_mean - 3*CM_std, CM_mean + 3*CM_std]
        plt.imshow(Map, origin='lower', extent=(0, len_x, 0, len_y))
        plt.clim(CM_mean - 3*CM_std, CM_mean + 3*CM_std)
        # cbar = plt.colorbar()
        # cbar.ax.set_ylabel('dI/dV(a.u.)',fontname='Arial',fontsize=16)
        # cbar.set_ticks(fontname='Arial',fontsize=12)
        plt.xlabel('$X$ (nm)', fontname='Arial', fontsize=12)
        plt.ylabel('$Y$ (nm)', fontname='Arial', fontsize=12)
        ax = plt.gca()
        plt.xticks(fontname='Arial', fontsize=12)
        plt.yticks(fontname='Arial', fontsize=12)
        # plt.xticks(np.round(np.linspace(ax.get_xlim()[0], ax.get_xlim()[1], 5), decimals=0), fontname='Arial', fontsize=12)
        # plt.yticks(np.round(np.linspace(ax.get_ylim()[0], ax.get_ylim()[1], 5), decimals=0), fontname='Arial', fontsize=12)
        ax.spines['bottom'].set_linewidth(1.5)
        ax.spines['top'].set_linewidth(1.5)
        ax.spines['left'].set_linewidth(1.5)
        ax.spines['right'].set_linewidth(1.5)
        # ax.xaxis.set_minor_locator(AutoMinorLocator())
        # ax.tick_params(width=2)
        ax.tick_params(axis="x", length=5, width=1.5, direction="in")
        ax.tick_params(axis="y", length=5, width=1.5, direction="in")
        # ax.tick_params(which='minor', length=3, width=2, direction="out")
        plt.tight_layout()


class AreaDetectionApp(QtWidgets.QMainWindow):
    def __init__(self):
        QtWidgets.QMainWindow.__init__(self)
        # self.ui = uic.loadUi('Image_hunting.ui', self)
        self.ui = Ui_MainWindow()
        self.ui.setupUi(self)
        # self.resize(888, 600)
        self.setWindowTitle('Area Detection Application')
        icon = QtGui.QIcon()
        icon.addPixmap(QtGui.QPixmap('icon.ico'), QtGui.QIcon.Normal, QtGui.QIcon.Off)
        self.setWindowIcon(icon)
        self.threadpool = QtCore.QThreadPool()
        self.matchingAlgor_list = ['TM_CCOEFF_NORMED (default)', 'TM_CCOEFF', 'TM_CCORR_NORMED', 'TM_CCORR']
        self.ui.comboBox_Alg.addItems(self.matchingAlgor_list)
        self.display_list = ['All', 'Highest Similarity']
        self.ui.comboBox_display.addItems(self.display_list)
        self.ui.comboBox_display.currentIndexChanged.connect(self.CBdisplay)
        self.fname1 = ''
        self.fname2 = ''
        self.ui.spinBox_threshold.setRange(0, 100)
        self.ui.spinBox_threshold.setValue(50)
        self.ui.LarImg_browse_button.clicked.connect(lambda: self.openFileNameDialog(1))
        self.ui.SmImg_browse_button.clicked.connect(lambda: self.openFileNameDialog(2))
        self.ui.pushButton.clicked.connect(self.cv2matching)
        self.ui.actionExit.triggered.connect(self.fileQuit)
        self.ui.actionAbout.triggered.connect(self.about)

    def openFileNameDialog(self, img_switch):
        options = QFileDialog.Options()
        # options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self, 'Open File', '', 'Nanonis SXM Files (*.sxm);;All Files (*)', options=options)
        if fileName == '':
            return
        fname = fileName.split('/')[-1]

        if not fname.endswith('.sxm'):
            self.wrongFileMsg()
            return

        img, len_x, len_y = cal.return_img(fileName)
        self.canvas = staticplot(self, width=5, height=5, dpi=100)
        self.canvas.plotImg(img, len_x, len_y)
        if img_switch == 1:
            self.fname1 = fileName
            self.ui.gridLayout_8.addWidget(self.canvas, 3, 1, 1, 1)
            self.ui.LarImg_edit.setText(str(fname))
        if img_switch == 2:
            self.fname2 = fileName
            self.ui.gridLayout_8.addWidget(self.canvas, 3, 3, 1, 1)
            self.ui.SmImg_edit.setText(str(fname))

    def cv2matching(self):
        if self.fname1 == '' or self.fname2 == '':
            self.emptyFileMsg()
            return
        threshold = self.ui.spinBox_threshold.value()
        Matching_algor = self.ui.comboBox_Alg.currentText()
        CBdisplay = self.ui.comboBox_display.currentText()

        returned_status = cal.matching_cal(self.fname1, self.fname2, display=CBdisplay,
                                                           threshold=threshold/100, Matching_algor=Matching_algor)
        if returned_status == 1:
            self.tooManyMatchMsg()
        return

    def CBdisplay(self):
        if self.ui.comboBox_display.currentText() == 'Highest Similarity':
            self.ui.spinBox_threshold.setEnabled(False)
        elif self.ui.comboBox_display.currentText() == 'All':
            self.ui.spinBox_threshold.setEnabled(True)

    def wrongFileMsg(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Error!')
        msg.setText('Wrong file type! Please choose a Nanonis SXM (.sxm) file.')
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.exec_()

    def emptyFileMsg(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Error!')
        msg.setText('Please choose two files for object detection!')
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.exec_()

    def tooManyMatchMsg(self):
        msg = QtWidgets.QMessageBox()
        msg.setWindowTitle('Error!')
        msg.setText('Too many detection results! Please use a higher similarity threshold.')
        msg.setIcon(QtWidgets.QMessageBox.Critical)
        msg.exec_()

    def fileQuit(self):
        self.close()

    def about(self):
        QtWidgets.QMessageBox.about(self, 'About', '''Copyright 2021 Junxiang(Eric) JIA.

This program uses OpenCV-Python (feature matching and object detection) to search and find the location of a small scale topography image in a larger one.

It may be used and modified with no restriction; raw copies as well as modified versions may be distributed without limitation.''')


app = QtWidgets.QApplication(sys.argv)
mainWindow = AreaDetectionApp()
mainWindow.show()
sys.exit(app.exec_())
