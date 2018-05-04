# -*- coding: utf-8 -*-

#Project submission of Megh Shukla : 173310008  and Ushasi Chaudhari : 174310003
#Project demonstrated on 2nd May, VIP Lab at 10pm to Deepak Sir
#Project title : GPU acceleration for Neural Networks using CUDA PYTHON

#Parts of this file is created using : PyQt4 UI code generator 4.11, from original UI file created in QtDesigner.

from PyQt4 import QtCore, QtGui
import ANN_gpu                                                                #Import to the main code file
import os
import sys
from matplotlib import pyplot as plt


#textInfo is what is displayed in the GUI Text Field box. Helps make the GUI user friendly.
textInfo='' 

#When we are running as executable, it is in frozen state, hence pathname needs to be executable location and not file location
#Lines 17 to 20 have been referred from stackoverflow.
if getattr(sys, 'frozen', False):
    os.chdir(os.path.dirname(sys.executable))
elif __file__:
    os.chdir(os.path.dirname(__file__))


#PyQt generated code
try:
    _fromUtf8 = QtCore.QString.fromUtf8
except AttributeError:
    def _fromUtf8(s):
        return s

try:
    _encoding = QtGui.QApplication.UnicodeUTF8
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig, _encoding)
except AttributeError:
    def _translate(context, text, disambig):
        return QtGui.QApplication.translate(context, text, disambig)


#design and layout of the GUI
class Ui_MainWindow(object):
    def setupUi(self, MainWindow):
        QtGui.QApplication.setStyle(QtGui.QStyleFactory.create('Cleanlooks'))
        MainWindow.setObjectName(_fromUtf8("MainWindow"))
        MainWindow.resize(747, 598)
        self.centralwidget = QtGui.QWidget(MainWindow)
        self.centralwidget.setObjectName(_fromUtf8("centralwidget"))
        self.gridLayout = QtGui.QGridLayout(self.centralwidget)
        self.gridLayout.setObjectName(_fromUtf8("gridLayout"))
        
        
        MainWindow.setCentralWidget(self.centralwidget)
        self.menubar = QtGui.QMenuBar(MainWindow)
        self.menubar.setGeometry(QtCore.QRect(0, 0, 747, 26))
        self.menubar.setObjectName(_fromUtf8("menubar"))
        
        self.menuMenu = QtGui.QMenu(self.menubar)
        self.menuMenu.setObjectName(_fromUtf8("menuMenu"))
        MainWindow.setMenuBar(self.menubar)
        
        self.statusbar = QtGui.QStatusBar(MainWindow)
        self.statusbar.setObjectName(_fromUtf8("statusbar"))
        MainWindow.setStatusBar(self.statusbar)
        
        self.actionQuit = QtGui.QAction(MainWindow)
        self.actionQuit.setObjectName(_fromUtf8("actionQuit"))
        self.actionQuit.setShortcut("Ctrl+Q")
        self.actionQuit.setStatusTip('Leave The App')
        
        self.menuMenu.addAction(self.actionQuit)
        self.menubar.addAction(self.menuMenu.menuAction())
        
        
        self.DatasetLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.DatasetLineEdit.setObjectName(_fromUtf8("DatasetLineEdit"))
        self.gridLayout.addWidget(self.DatasetLineEdit, 0, 1, 2, 2)
        
        self.DatasetBrowse = QtGui.QPushButton(self.centralwidget)
        self.DatasetBrowse.setObjectName(_fromUtf8("DatasetBrowse"))
        self.gridLayout.addWidget(self.DatasetBrowse, 0, 3, 2, 1)
        self.DatasetBrowse.clicked.connect(self.getDataset)
        self.DatasetBrowse.setStatusTip('Browse for Dataset')
        
        spacerItem = QtGui.QSpacerItem(20, 31, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem, 1, 2, 1, 1)
        
        self.NumberOfSamplesLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.NumberOfSamplesLineEdit.setObjectName(_fromUtf8("NumberOfSamplesLineEdit"))
        self.gridLayout.addWidget(self.NumberOfSamplesLineEdit, 2, 1, 2, 2)
        self.NumberOfSamplesLineEdit.setStatusTip('Enter number of samples to randomly choose from dataset')
        
        self.Example = QtGui.QPushButton(self.centralwidget)
        self.Example.setObjectName(_fromUtf8("Example"))
        self.gridLayout.addWidget(self.Example, 2, 3, 2, 1)
        self.Example.setStatusTip('Click to view example')
        self.Example.clicked.connect(self.ExampleClick)
        
        spacerItem1 = QtGui.QSpacerItem(20, 30, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem1, 3, 2, 1, 1)
        
        self.TrainTestValLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.TrainTestValLineEdit.setObjectName(_fromUtf8("TrainTestValLineEdit"))
        self.gridLayout.addWidget(self.TrainTestValLineEdit, 4, 1, 2, 2)
        self.TrainTestValLineEdit.setStatusTip('Enter training validation and testing split')
        
        spacerItem2 = QtGui.QSpacerItem(20, 31, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem2, 5, 2, 1, 1)
        
        self.NetworkArchitectureLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.NetworkArchitectureLineEdit.setObjectName(_fromUtf8("NetworkArchitectureLineEdit"))
        self.gridLayout.addWidget(self.NetworkArchitectureLineEdit, 6, 1, 2, 2)
        self.NetworkArchitectureLineEdit.setStatusTip('Enter number of neurons in input, hidden(s) and ouput layer')
        
        spacerItem3 = QtGui.QSpacerItem(20, 30, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem3, 7, 2, 1, 1)
        
        self.NetworkHyperparametersLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.NetworkHyperparametersLineEdit.setObjectName(_fromUtf8("NetworkHyperparametersLineEdit"))
        self.gridLayout.addWidget(self.NetworkHyperparametersLineEdit, 8, 1, 2, 2)
        self.NetworkHyperparametersLineEdit.setStatusTip('Enter sigmoidal gain, threshold, learning rate, momentum factor and regularization factor')
        
        spacerItem4 = QtGui.QSpacerItem(20, 31, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem4, 9, 2, 1, 1)
        
        self.MaximumEpochsLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.MaximumEpochsLabel.setFont(font)
        self.MaximumEpochsLabel.setObjectName(_fromUtf8("MaximumEpochsLabel"))
        self.gridLayout.addWidget(self.MaximumEpochsLabel, 10, 0, 2, 1)
        
        self.MaximumEpochsLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.MaximumEpochsLineEdit.setObjectName(_fromUtf8("MaximumEpochsLineEdit"))
        self.gridLayout.addWidget(self.MaximumEpochsLineEdit, 10, 1, 2, 2)
        self.MaximumEpochsLineEdit.setStatusTip('Max limit on epochs')
        
        spacerItem5 = QtGui.QSpacerItem(20, 30, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem5, 11, 2, 1, 1)
        
        self.SaveLocationLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.SaveLocationLabel.setFont(font)
        self.SaveLocationLabel.setObjectName(_fromUtf8("SaveLocationLabel"))
        self.gridLayout.addWidget(self.SaveLocationLabel, 12, 0, 2, 1)
        
        self.SaveLocationLineEdit = QtGui.QLineEdit(self.centralwidget)
        self.SaveLocationLineEdit.setObjectName(_fromUtf8("SaveLocationLineEdit"))
        self.gridLayout.addWidget(self.SaveLocationLineEdit, 12, 1, 2, 2)
        self.SaveLocationLineEdit.setStatusTip('See browse')
        
        self.Run = QtGui.QPushButton(self.centralwidget)
        self.Run.setObjectName(_fromUtf8("Run"))
        self.gridLayout.addWidget(self.Run, 12, 3, 2, 1)
        self.Run.clicked.connect(self.RunClick)
        
        spacerItem6 = QtGui.QSpacerItem(20, 31, QtGui.QSizePolicy.Minimum, QtGui.QSizePolicy.Expanding)
        self.gridLayout.addItem(spacerItem6, 13, 2, 1, 1)
        
        self.ConsolueOutput = QtGui.QTextEdit(self.centralwidget)
        self.ConsolueOutput.setObjectName(_fromUtf8("ConsolueOutput"))
        self.gridLayout.addWidget(self.ConsolueOutput, 14, 0, 1, 4)
        
        self.DatasetLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.DatasetLabel.setFont(font)
        self.DatasetLabel.setObjectName(_fromUtf8("DatasetLabel"))
        self.gridLayout.addWidget(self.DatasetLabel, 1, 0, 1, 1)
        
        self.NumberOfSamplesLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.NumberOfSamplesLabel.setFont(font)
        self.NumberOfSamplesLabel.setObjectName(_fromUtf8("NumberOfSamplesLabel"))
        self.gridLayout.addWidget(self.NumberOfSamplesLabel, 3, 0, 1, 1)
        
        self.TrainTestValLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.TrainTestValLabel.setFont(font)
        self.TrainTestValLabel.setObjectName(_fromUtf8("TrainTestValLabel"))
        self.gridLayout.addWidget(self.TrainTestValLabel, 5, 0, 1, 1)
        
        self.NetworkArchitectureLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.NetworkArchitectureLabel.setFont(font)
        self.NetworkArchitectureLabel.setObjectName(_fromUtf8("NetworkArchitectureLabel"))
        self.gridLayout.addWidget(self.NetworkArchitectureLabel, 7, 0, 1, 1)
        
        self.NetworkHyperparametersLabel = QtGui.QLabel(self.centralwidget)
        font = QtGui.QFont()
        font.setFamily(_fromUtf8("Calibri"))
        font.setPointSize(10)
        font.setItalic(False)
        self.NetworkHyperparametersLabel.setFont(font)
        self.NetworkHyperparametersLabel.setObjectName(_fromUtf8("NetworkHyperparametersLabel"))
        self.gridLayout.addWidget(self.NetworkHyperparametersLabel, 9, 0, 1, 1)
        
        self.actionQuit.triggered.connect(self.closeApplication)
        self.retranslateUi(MainWindow)
        QtCore.QMetaObject.connectSlotsByName(MainWindow)

    def retranslateUi(self, MainWindow):
        MainWindow.setWindowTitle(_translate("MainWindow", "Neural Network EE769", None))
        MainWindow.setWindowIcon(QtGui.QIcon('.\IITB.png'))
        
        self.DatasetBrowse.setText(_translate("MainWindow", "Browse", None))
        self.Example.setText(_translate("MainWindow", "Example", None))
        self.MaximumEpochsLabel.setText(_translate("MainWindow", "Maximum Epochs :", None))
        self.SaveLocationLabel.setText(_translate("MainWindow", "Save Location : ", None))
        self.Run.setText(_translate("MainWindow", "Browse save and Run!", None))
        self.DatasetLabel.setText(_translate("MainWindow", "Dataset : ", None))
        self.NumberOfSamplesLabel.setText(_translate("MainWindow", "Number of Samples : ", None))
        self.TrainTestValLabel.setText(_translate("MainWindow", "Train / Validation / Test : ", None))
        self.NetworkArchitectureLabel.setText(_translate("MainWindow", "Network Architecture : ", None))
        self.NetworkHyperparametersLabel.setText(_translate("MainWindow", "Network Hyperparameters :", None))
        self.menuMenu.setTitle(_translate("MainWindow", "Menu", None))
        self.actionQuit.setText(_translate("MainWindow", "Quit", None))
        
        
    
    def ExampleClick(self):
        '''
        A message box that pops up with steps on using the GUI
        '''
        QtGui.QMessageBox.information(MainWindow, 'Help', "Steps to follow:\n1. Number of Samples: 1200\n\n2. Train / Validation / Test: [0.6,0.3,0.1]\n\n3. Network Architecture: [8,4,2,1]\n\n4. Network Hyperparameters: [1,0,0.0001,0,0.005]\n\n5. Maximum Epochs: 25")
    
    
    def getDataset(self):
        '''
        Identify directory of Dataset
        '''
        global DirectoryDataset
        global textInfo    
        self.DatasetDirectory = QtGui.QFileDialog.getOpenFileName(MainWindow, 'Open File',os.getcwd(),'*.csv')
        self.DatasetLineEdit.setText(str(self.DatasetDirectory))
        textInfo+=('Loaded Dataset\n')
        self.ConsolueOutput.setText(textInfo)
        
    
    def RunClick(self):
        '''
        Checks if all the entries provided in the field are correct or not. Example: Number of epochs should be int. Any other data type is rejected.
        After all entries are checked for correct data type, calls the script function in ANN_gpu, which does the actual evaluation of the Neural Network.
        Taken return value of execution time, testing error and displays in the GUI
        '''
        global textInfo
        #Variables to pass: N_Samples, TrainTestValidation, NetArch, Hyperparams, MaxEpochs, SaveLocation
        #Checks if all entries have valid data types
        try:
            eval(self.NumberOfSamplesLineEdit.text()) 
            if(type(eval(self.NumberOfSamplesLineEdit.text()))!=int):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not Integer", QtGui.QMessageBox.Ok) 
                self.NumberOfSamplesLineEdit.setText('')    
            else:
                self.N_Samples=eval(self.NumberOfSamplesLineEdit.text())
                textInfo+=('Number of Samples : '+str(self.N_Samples)+'\n')
                self.ConsolueOutput.setText(textInfo)
            eval(self.TrainTestValLineEdit.text())
            if(type(eval(self.TrainTestValLineEdit.text()))!=list):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not List", QtGui.QMessageBox.Ok) 
                self.TrainTestValLineEdit.setText('')    
            else: 
                self.TrainTestValidation=eval(self.TrainTestValLineEdit.text())
                textInfo+=('Training, Validation, Testing ratio : '+str(self.TrainTestValidation)+'\n')
                self.ConsolueOutput.setText(textInfo)
            eval(self.NetworkArchitectureLineEdit.text())
            if(type(eval(self.NetworkArchitectureLineEdit.text()))!=list):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not List", QtGui.QMessageBox.Ok) 
                self.NetworkArchitectureLineEdit.setText('')    
            else: 
                self.NetArch=eval(self.NetworkArchitectureLineEdit.text())
                textInfo+=('Network Architecture : '+str(self.NetArch)+'\n')
                self.ConsolueOutput.setText(textInfo)
            eval(self.NetworkHyperparametersLineEdit.text())
            if(type(eval(self.NetworkHyperparametersLineEdit.text()))!=list):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not List", QtGui.QMessageBox.Ok) 
                self.NetworkHyperparametersLineEdit.setText('')    
            else: 
                self.Hyperparams=eval(self.NetworkHyperparametersLineEdit.text())
                textInfo+=('Sigmoidal gain, Threshold, Learning rate, Momentum factor and Regularization factor  : '+str(self.Hyperparams)+'\n')
                self.ConsolueOutput.setText(textInfo)
            eval(self.MaximumEpochsLineEdit.text())
            if(type(eval(self.MaximumEpochsLineEdit.text()))!=int):
                QtGui.QMessageBox.information(MainWindow, 'Incorrect Input', "Input type not Integer", QtGui.QMessageBox.Ok) 
                self.MaximumEpochsLineEdit.setText('')    
            else: 
                self.MaxEpochs=eval(self.MaximumEpochsLineEdit.text())
                textInfo+=('Maximum Epoch Count  : '+str(self.MaxEpochs)+'\n')
                self.ConsolueOutput.setText(textInfo)
            self.SaveLocation = QtGui.QFileDialog.getExistingDirectory(MainWindow, 'Select Save Location',os.getcwd())
            self.SaveLocation+="\\"
            self.SaveLocationLineEdit.setText(str(self.SaveLocation))
            textInfo+=('Performing Neural Network Classification, See Console for details\n')
            self.ConsolueOutput.setText(textInfo)
            
        
        except:
            QtGui.QMessageBox.information(MainWindow, 'Invalid Input', "Check input formatting", QtGui.QMessageBox.Ok)
        
        #Calls the script function in ANN_gpu
        try:
            [self.TestError,self.Time]=ANN_gpu.script(self.N_Samples,self.TrainTestValidation,self.NetArch,self.Hyperparams,self.MaxEpochs,self.SaveLocation, self.DatasetDirectory)
            textInfo+=('\nError on Test Dataset : '+str(self.TestError)+'\n')
            self.ConsolueOutput.setText(textInfo)
            textInfo+=('Saved the model as FinalWeights.pkl\n')
            self.ConsolueOutput.setText(textInfo)
            textInfo+=('Time taken per iteration : %f seconds\n'%(self.Time/(self.N_Samples*self.TrainTestValidation[0]*self.MaxEpochs)))
            self.ConsolueOutput.setText(textInfo)
            plt.show()
        except: 
            print("Unexpected error:", sys.exc_info()[0])
            QtGui.QMessageBox.information(MainWindow, 'Invalid Values', "Could not perform Neural Network Classification, recheck user input", QtGui.QMessageBox.Ok)
    
    def closeApplication(self):
        '''
        Terminates the application.
        '''
        self.choice = QtGui.QMessageBox.question(MainWindow, 'Exit', "Are you sure you want to exit?", QtGui.QMessageBox.Yes | QtGui.QMessageBox.No)
        if self.choice == QtGui.QMessageBox.Yes:
            print("Exiting...")
            sys.exit()
        else:
            pass 


if __name__ == "__main__":
    import sys
    app = QtGui.QApplication(sys.argv)
    MainWindow = QtGui.QMainWindow()
    ui = Ui_MainWindow()
    ui.setupUi(MainWindow)
    MainWindow.show()
    sys.exit(app.exec_())

