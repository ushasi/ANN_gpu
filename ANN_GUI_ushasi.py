# -*- coding: utf-8 -*-

# Form implementation generated from reading ui file 'ANN_GUI_ushasi.ui'
#
# Created by: PyQt5 UI code generator 5.6
#
# WARNING! All changes made in this file will be lost!

from PyQt5 import QtCore, QtGui, QtWidgets

class Ui_Dialog(object):
    def setupUi(self, Dialog):
        Dialog.setObjectName("Dialog")
        Dialog.resize(563, 486)
        self.buttonBox = QtWidgets.QDialogButtonBox(Dialog)
        self.buttonBox.setGeometry(QtCore.QRect(30, 420, 341, 32))
        self.buttonBox.setOrientation(QtCore.Qt.Horizontal)
        self.buttonBox.setStandardButtons(QtWidgets.QDialogButtonBox.Cancel|QtWidgets.QDialogButtonBox.Ok)
        self.buttonBox.setObjectName("buttonBox")
        self.BrowseButton = QtWidgets.QPushButton(Dialog)
        self.BrowseButton.setGeometry(QtCore.QRect(450, 40, 75, 31))
        self.BrowseButton.setCheckable(False)
        self.BrowseButton.setFlat(False)
        self.BrowseButton.setObjectName("BrowseButton")
        self.textEdit = QtWidgets.QTextEdit(Dialog)
        self.textEdit.setGeometry(QtCore.QRect(20, 40, 401, 31))
        self.textEdit.setObjectName("textEdit")
        self.graphicsView = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView.setGeometry(QtCore.QRect(10, 170, 256, 192))
        self.graphicsView.setObjectName("graphicsView")
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(290, 170, 256, 192))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.groupBox = QtWidgets.QGroupBox(Dialog)
        self.groupBox.setGeometry(QtCore.QRect(0, 150, 271, 221))
        self.groupBox.setFlat(False)
        self.groupBox.setCheckable(False)
        self.groupBox.setObjectName("groupBox")
        self.graphicsView.raise_()
        self.groupBox_2 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_2.setGeometry(QtCore.QRect(280, 150, 281, 221))
        self.groupBox_2.setObjectName("groupBox_2")
        self.graphicsView_2.raise_()
        self.groupBox_3 = QtWidgets.QGroupBox(Dialog)
        self.groupBox_3.setGeometry(QtCore.QRect(10, 20, 531, 61))
        self.groupBox_3.setObjectName("groupBox_3")
        self.comboBox = QtWidgets.QComboBox(Dialog)
        self.comboBox.setGeometry(QtCore.QRect(220, 110, 131, 22))
        self.comboBox.setMaxVisibleItems(2)
        self.comboBox.setMaxCount(2147483646)
        self.comboBox.setMinimumContentsLength(2)
        self.comboBox.setModelColumn(2)
        self.comboBox.setObjectName("comboBox")
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(20, 380, 256, 192))
        self.tableWidget.setObjectName("tableWidget")
        self.tableWidget.setColumnCount(0)
        self.tableWidget.setRowCount(0)

        self.retranslateUi(Dialog)
        self.comboBox.setCurrentIndex(-1)
        self.buttonBox.accepted.connect(Dialog.accept)
        self.buttonBox.rejected.connect(Dialog.reject)
        QtCore.QMetaObject.connectSlotsByName(Dialog)
        Dialog.setTabOrder(self.textEdit, self.BrowseButton)
        Dialog.setTabOrder(self.BrowseButton, self.graphicsView_2)
        Dialog.setTabOrder(self.graphicsView_2, self.graphicsView)
        Dialog.setTabOrder(self.graphicsView, self.comboBox)

    def retranslateUi(self, Dialog):
        _translate = QtCore.QCoreApplication.translate
        Dialog.setWindowTitle(_translate("Dialog", "Dialog"))
        self.BrowseButton.setText(_translate("Dialog", "Browse File"))
        self.groupBox.setTitle(_translate("Dialog", "Error Graph"))
        self.groupBox_2.setTitle(_translate("Dialog", "Validation Graph"))
        self.groupBox_3.setTitle(_translate("Dialog", "File Name"))


if __name__ == "__main__":
    import sys
    app = QtWidgets.QApplication(sys.argv)
    Dialog = QtWidgets.QDialog()
    ui = Ui_Dialog()
    ui.setupUi(Dialog)
    Dialog.show()
    sys.exit(app.exec_())

