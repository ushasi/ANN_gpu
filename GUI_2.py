# -*- coding: utf-8 -*-
"""
Created on Sun Apr 29 22:57:42 2018

@author: USHASI
"""

import sys
from PyQt5.QtWidgets import QApplication, QWidget, QInputDialog, QLineEdit,QPushButton,QFileDialog,QLabel,QTableWidget,QTableWidgetItem,QVBoxLayout
from PyQt5.QtGui import QIcon, QPixmap
from PyQt5.QtCore import pyqtSlot

#import ANN_gpu
 
class App(QWidget):
 
    def __init__(self):
        super().__init__()
        self.title = 'PyQt5 input dialogs - pythonspot.com'
        self.left = 100
        self.top = 100
        self.width = 400
        self.height = 400
        self.initUI()
 
    def initUI(self):
        self.setWindowTitle(self.title)
        self.setGeometry(self.left, self.top, self.width, self.height)
 
        
        button = QPushButton('Browse File', self)
        button.setToolTip('This is an example button')
        button.move(100,70) 
        button.clicked.connect(self.openFileNameDialog)
        
        # Create textbox
        self.textbox = QLineEdit(self)
        self.textbox.move(20, 20)
        self.textbox.resize(280,40)
        
        
        
        self.getSamples()
        self.getSplit()
        self.getInput()
        self.getHidden()
        self.getOutput()
        self.getGain()
        self.getThreshold()
        self.getLRate()
        self.getMF()
        self.getRF()
        
        #Call the code here
        #label = QLabel(self)
        #pixmap = QPixmap('cube_dir.png') #Show the Error Graph File here
        #label.setPixmap(pixmap)
        #self.resize(pixmap.width(),pixmap.height())
        
        self.createTable()
        
        button2 = QPushButton('View Table', self)
        button2.setToolTip('This is an example button')
        button2.move(100,200) 
        button2.clicked.connect(self.openLayout)
        
        self.graphicsView_2 = QtWidgets.QGraphicsView(Dialog)
        self.graphicsView_2.setGeometry(QtCore.QRect(290, 170, 256, 192))
        self.graphicsView_2.setObjectName("graphicsView_2")
        self.graphicsView_2.raise_()
 
        # Add box layout, add table to box layout and add box layout to widget
         
        
        self.show()
        
    def createTable(self):
       # Create table
        self.tableWidget = QTableWidget()
        self.tableWidget.setRowCount(10)
        self.tableWidget.setColumnCount(2)
        self.tableWidget.setItem(0,0, QTableWidgetItem("No of Samples"))
        self.tableWidget.setItem(0,1, QTableWidgetItem("Cell (1,2)"))
        self.tableWidget.setItem(1,0, QTableWidgetItem("Split"))
        self.tableWidget.setItem(1,1, QTableWidgetItem("Cell (2,2)"))
        self.tableWidget.setItem(2,0, QTableWidgetItem("Nodes in Input Layer"))
        self.tableWidget.setItem(2,1, QTableWidgetItem("Cell (3,2)"))
        self.tableWidget.setItem(3,0, QTableWidgetItem("Nodes in Hidden Layer"))
        self.tableWidget.setItem(3,1, QTableWidgetItem("Cell (4,2)"))
        self.tableWidget.setItem(4,0, QTableWidgetItem("Nodes in Output Layer"))
        self.tableWidget.setItem(4,1, QTableWidgetItem("Cell (5,2)"))
        self.tableWidget.setItem(5,0, QTableWidgetItem("Sigmoidal Gain"))
        self.tableWidget.setItem(5,1, QTableWidgetItem("Cell (6,2)"))
        self.tableWidget.setItem(6,0, QTableWidgetItem("Threshold"))
        self.tableWidget.setItem(6,1, QTableWidgetItem("Cell (7,2)"))
        self.tableWidget.setItem(7,0, QTableWidgetItem("Learning Rate"))
        self.tableWidget.setItem(7,1, QTableWidgetItem("Cell (8,2)"))
        self.tableWidget.setItem(8,0, QTableWidgetItem("Momentum Factor"))
        self.tableWidget.setItem(8,1, QTableWidgetItem("Cell (9,2)"))
        self.tableWidget.setItem(9,0, QTableWidgetItem("Regularization Factor"))
        self.tableWidget.setItem(9,1, QTableWidgetItem("Cell (10,2)"))
        self.tableWidget.move(0,0)
 
        # table selection change
        self.tableWidget.doubleClicked.connect(self.on_clickTable)
        
    @pyqtSlot()
    def on_clickTable(self):
        print("\n")
        for currentQTableWidgetItem in self.tableWidget.selectedItems():
            print(currentQTableWidgetItem.row(), currentQTableWidgetItem.column(), currentQTableWidgetItem.text())

    @pyqtSlot()
    def on_click(self):
        print('PyQt5 button click')
    @pyqtSlot()   
    def openFileNameDialog(self):    
        options = QFileDialog.Options()
        options |= QFileDialog.DontUseNativeDialog
        fileName, _ = QFileDialog.getOpenFileName(self,"QFileDialog.getOpenFileName()", "","All Files (*);;Python Files (*.py)", options=options)
        if fileName:
            print(fileName)
            textboxValue = fileName
            self.textbox.setText(textboxValue)  
            
    @pyqtSlot()   
    def openLayout(self): 
        self.tableWidget = QtWidgets.QTableWidget(Dialog)
        self.tableWidget.setGeometry(QtCore.QRect(20, 380, 256, 192))
        self.tableWidget.setObjectName("tableWidget")
        '''
        self.layout = QVBoxLayout()
        self.layout.addWidget(self.tableWidget) 
        self.setLayout(self.layout)
        '''
        
 
    def getSamples(self):
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter number of data samples:", 28, 0, 100, 1)
        if okPressed:
            print(i)           
    def getSplit(self):
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter training, validation, testing split:", 28, 0, 100, 1)
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter training, validation, testing split:", 28, 0, 100, 1)
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter training, validation, testing split:", 28, 0, 100, 1)
        if okPressed:
            print(i)           
    def getInput(self):
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter number of nodes in Input layer:", 28, 0, 100, 1)
        if okPressed:
            print(i)            
    def getHidden(self):
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter number of nodes in Hidden layer:", 28, 0, 100, 1)
        if okPressed:
            print(i)            
    def getOutput(self):
        i, okPressed = QInputDialog.getInt(self, "Get integer","Enter number of nodes in Output layer:", 28, 0, 100, 1)
        if okPressed:
            print(i) 
    def getGain(self):
        d, okPressed = QInputDialog.getDouble(self, "Get double","Sigmoidal Gain:", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
    def getThreshold(self):
        d, okPressed = QInputDialog.getDouble(self, "Get double","Threshold:", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
    def getLRate(self):
        d, okPressed = QInputDialog.getDouble(self, "Get double","Learning Rate:", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
    def getMF(self):
        d, okPressed = QInputDialog.getDouble(self, "Get double","Momentum Factor:", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
    def getRF(self):
        d, okPressed = QInputDialog.getDouble(self, "Get double","Regularization Factor", 10.50, 0, 100, 10)
        if okPressed:
            print( d)
   
 
 
if __name__ == '__main__':
    app = 0
    app = QApplication(sys.argv)
    ex = App()
    sys.exit(app.exec_())