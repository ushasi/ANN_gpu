#To see the submission demonstration, run ANNGui, which executes the GUI and makes calls to functions written here.
#Project submission of Megh Shukla : 173310008  and Ushasi Chaudhari : 174310003
#Project demonstrated on 2nd May, VIP Lab at 10pm to Deepak Sir
#Project title : GPU acceleration for Neural Networks using CUDA PYTHON

#Backpropagation algorithm referred from Neural Networks and Learning Machines, Simon Haykin


import time                                                                    #Time import to calculate time taken for execution of FeedForward and Backpropagation
import math                                                                    #Basic math functions and constants
import numpy as np                                                            #Only use of numpy is representation of Matrix. No numpy multiplication or transpose is used
import pandas as pd                                                           #Handling of the dataset
from numba import cuda                                                        #JIT compiler for cuda kernels, handles essence of the code
from sklearn import preprocessing                                             #Only use is scaling the data. 
                                                                                #We used sklearn preprocessing because we already covered preprocessing code in first assignment
from matplotlib import pyplot as plt                                          #Generate graphs 
import pickle                                                                  #Saving the model 

exp=math.e
EpochMax=0

#Global Variables that store error per iteration (1 forward pass + 1 backward pass) and per epoch (1 pass of the entire dataset)
ErrorPerIterationTrain=[]
ErrorPerEpochTrain=[]
ErrorPerIterationValid=[]
ErrorPerEpochValid=[]
ErrorPerIterationTest=[]
ErrorPerEpochTest=[]

NumberOfSamples=0

######################################################################### CUDA KERNEL ##################################################################

#cuda kernel defined by cuda.reduce(sum_reduce)
def sum_reduce(a, b):
    '''
    To calculate sum values across each thread launched by the kernel
    '''
    return a+b
    
    
@cuda.jit
def transpose2_matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B'
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[col, k]
        C[row, col] = tmp    
    
    

@cuda.jit    
def transpose_matmul(A, B, C):                     #A is the transposed element
    """
    Perform matrix multiplication of C = A' * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[0]):
            tmp += A[k, row] * B[k, col]
        C[row, col] = tmp


#This kernel (matmul is sourced from the internet). It's very intuitive and simple.
#Modifications to matmul (transpose versions above) are written by us.
@cuda.jit
def matmul(A, B, C):
    """
    Perform matrix multiplication of C = A * B
    """
    row, col = cuda.grid(2)
    if row < C.shape[0] and col < C.shape[1]:
        tmp = 0.
        for k in range(A.shape[1]):
            tmp += A[row, k] * B[k, col]
        C[row, col] = tmp

@cuda.jit()
def DeltaHidden(NodeOutputValues, temp, NodeDeltaValues):
    '''
    Computes the sensitivity of all neurons in the hidden layer
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
    
    if(i<NodeOutputValues.shape[0] and j<NodeOutputValues.shape[1]):
        NodeDeltaValues[i,j]=2*(NodeOutputValues[i,j])*(1-NodeOutputValues[i,j])*temp[j,i+1]
    else:
        return



@cuda.jit()
def DeltaOutput(target,NodeOutputValues, NodeDeltaValues, networkParameters):
    '''
    Computes the sensitivity of all neurons in the output layer
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
    
    if(i<NodeOutputValues.shape[0] and j<NodeOutputValues.shape[1]):
        NodeDeltaValues[i,j]=networkParameters[0]*(target[i,j]-NodeOutputValues[i,j])
    else:
        return
    

@cuda.jit()
def Sigmoid(NodeInputValues,NodeOutputValues, networkParameters):
    '''
    Sigmoid activation function
    input = network hyperparameters
    returns activated value in -1 to 1 
    
    PLEASE NOTE : Sigmoid is scaled from -1 to 1 and not the original 0 to 1.
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
 
    if(i<NodeOutputValues.shape[0] and j<NodeOutputValues.shape[1]):
        NodeOutputValues[i,j]=(2*(1.0/(1+pow(exp,-networkParameters[0]*(NodeInputValues[i,j]-networkParameters[1])))))-1
    else:
        return
    


@cuda.jit()
def ErrorCalculate(error, NetworkOutput, Target):
    '''
    Calculates the error per output neuron, loss function : Mean Square Error
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
    
    if(i<NetworkOutput.shape[0] and j<NetworkOutput.shape[1]):
        error[i,j]=(Target[i,j]-NetworkOutput[i,j])**2
    else:
        return
    
    
@cuda.jit()
def deltaWeight(weightChange,learnRate, old_weightChange, momentum, regularization,weightOld):
    '''
    Change in weights in the present iteration, taking into account learning rate, momentum and regularization
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
    
    if(i<weightChange.shape[0] and j<weightChange.shape[1]):
        weightChange[i,j]=(learnRate*weightChange[i,j])+(momentum*old_weightChange[i,j])-(learnRate*regularization*weightOld[i,j])
    else:
        return  
        
@cuda.jit()                  
def weightUpdate(weights, weightChange):  
    '''
    New weights post backpropagation
    '''        
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy      
    
    if(i<weights.shape[0] and j<weights.shape[1]):
        weights[i,j]=weights[i,j]+weightChange[i,j]
    else:
        return      
                   
    
@cuda.jit()
def copy(node, node_withOne):
    '''
    Copies value from output to output which includes the bias neuron
    Two matrices, one with and the other without the bias were made because of matrix dimension issues that arise due to matrix multiplication.
    Input to layer j will be j_neurons x 1, but input to k layer will multiply j_neurons +1 * k_neurons weight matrix with j_neurons + 1 in jth layer 
    '''
    threadIDx=cuda.threadIdx.x
    threadIDy=cuda.threadIdx.y
    blockIDx=cuda.blockIdx.x
    blockIDy=cuda.blockIdx.y
    blockX=cuda.blockDim.x
    blockY=cuda.blockDim.y
    i=(blockX*blockIDx)+threadIDx
    j=(blockY*blockIDy)+threadIDy
    
    if(i<node.shape[0] and j<node.shape[1]):
        node_withOne[i+1,j]=node[i,j]
    else:
        return
            
################################################################## NEURAL NETWORK ARCHITECTURE ###################################################################
    
class NeuralNetwork(object):
    '''
    Defines the weight initialization and architecture of the network.
    Transfers the matrices from CPU memory to GPU memory
    Inputs : Nodes in layer, hyperparameters
    Output : delta matrices for all nodes (input layer delta unused)
             Weight initialization and storing previous weight change
             Node input and output value matrices
    '''
    def __init__(self,arraySize,networkParameters):
        self.outputNodes=arraySize[-1]
        self.inputNodes=arraySize[0]
        if(len(arraySize)>=3):
            self.hiddenNodes=arraySize[1:-1]
        
        #Hyperparametr initalization    
        self.Lambda=networkParameters[0]                                        #
        self.Threshold=networkParameters[1]                                     #init network parameters
        self.LearnRate=networkParameters[2]                                     #
        self.Alpha=networkParameters[3]                                         #
        self.Regularization=networkParameters[4]
        self.NodesByLayer=arraySize  
        
        #Making the scope of the variables global, these variables will be assigned to GPU memory  
        global device_InputIdentityMatrix
        global device_OutputIdentityMatrix
        global device_NodeDeltaValues
        global device_NodeOutputValues
        global device_NodeOutputValues_withOne
        global device_NodeInputValues
        global device_arraySize
        global device_networkParameters
        global device_Weights
        global device_PreviousWeightChange
        global device_WeightChange
                                
        #Weights initialization
        #These are copied to device memory from host memory 
        device_Weights=[0]*(len(self.NodesByLayer)-1)
        device_PreviousWeightChange=[0]*(len(self.NodesByLayer)-1)
        device_WeightChange=[0]*(len(self.NodesByLayer)-1)
        for i in range(len(self.NodesByLayer)-1):                                                    #random weights initialized ip : i nodes; hidden : j nodes; matrix : jxi
            device_Weights[i]=np.random.rand(self.NodesByLayer[i+1],self.NodesByLayer[i]+1)
            device_Weights[i]=cuda.to_device((np.matrix(device_Weights[i],dtype=np.float32)))
            device_PreviousWeightChange[i]=cuda.to_device(np.zeros((self.NodesByLayer[i+1],self.NodesByLayer[i]+1),dtype=np.float32))
            device_WeightChange[i]=cuda.to_device(np.zeros((self.NodesByLayer[i+1],self.NodesByLayer[i]+1),dtype=np.float32))
        
        #Values associated with the neurons in the layer 
        #These are copied to device memory from host memory 
        device_NodeOutputValues=[0]*len(self.NodesByLayer) 
        device_NodeOutputValues_withOne=[0]*len(self.NodesByLayer)
        device_NodeInputValues=[0]*len(self.NodesByLayer) 
        device_NodeDeltaValues=[0]*len(self.NodesByLayer)                         
        for i in range(len(self.NodesByLayer)):
            if(i!=len(self.NodesByLayer)-1):
                #Keeping space for the bias neuron "1"
                device_NodeOutputValues_withOne[i]=cuda.to_device(np.ones((self.NodesByLayer[i]+1,1),dtype=np.float32))
            else:
                #Output layer does not have bias neuron
                device_NodeOutputValues_withOne[i]=cuda.to_device(np.ones((self.NodesByLayer[i],1),dtype=np.float32))
            device_NodeOutputValues[i]=cuda.to_device(np.zeros((self.NodesByLayer[i],1),dtype=np.float32))                       #nx1 matrix
            device_NodeInputValues[i]=cuda.to_device(np.zeros((self.NodesByLayer[i],1),dtype=np.float32))
            device_NodeDeltaValues[i]=cuda.to_device(np.zeros((self.NodesByLayer[i],1),dtype=np.float32))
            
        InputIdentityMatrix=np.identity(self.NodesByLayer[0],dtype=np.float32)
        OutputIdentityMatrix=np.identity(self.NodesByLayer[-1],dtype=np.float32)    
             
        #Transferring to device
        device_InputIdentityMatrix=cuda.to_device(InputIdentityMatrix)
        device_OutputIdentityMatrix=cuda.to_device(OutputIdentityMatrix)
        device_arraySize=cuda.to_device(arraySize)
        device_networkParameters=cuda.to_device(networkParameters)

        
    def __str__(self):
        return 'ANN with input node :'+str(self.inputNodes)+'\nANN with output node :'+str(self.outputNodes)+'\nANN with hidden node s:'+str(self.hiddenNodes)
        

class Layers(object):
    '''
    Each layer is associated with it's number to identify it's location in the neural network
    Associated with two methods:
        input() : Computes the input to the present layer of neurons
        output(): Computes the output post activation of the neurons on the layer
    '''
    
    def __init__(self,layer_number):                                   
        self.layer_number=layer_number                                           
        
    def input(self,inData):
        #Layer 0 corresponds to input layer
        if(self.layer_number==0):
            #Input is passed on as is
            device_NodeInputValues[self.layer_number]=inData
        else: 
            #Output of the previous layer along with "1" is multplied by the weights associated with the previous layer to give Input to the present layer                                                                                             
            self.ThreadsPerBlockClassify=(16,16)                                                           
            self.bpgX=(device_Weights[self.layer_number-1].shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
            self.bpgY=(device_Weights[self.layer_number-1].shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
            self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
            '''Kernel call'''
            matmul[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_Weights[self.layer_number-1], device_NodeOutputValues_withOne[self.layer_number-1], device_NodeInputValues[self.layer_number])
            del self.bpgX,self.bpgY
            
            
    def output(self):
        #Layer 0 corresponds to input layer
        if(self.layer_number==0):
            #Output of 1st layer is same as the input                   
            self.ThreadsPerBlockClassify=(16,16)                                                            
            self.bpgX=(device_InputIdentityMatrix.shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
            self.bpgY=(device_InputIdentityMatrix.shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
            self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
            '''Kernel call'''
            matmul[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_InputIdentityMatrix, device_NodeInputValues[self.layer_number], device_NodeOutputValues[self.layer_number])
            del self.bpgX,self.bpgY
            
            #copy to new withOne array
            #We append 1 at the top of the output neuron array
            #This 1 takes care of the bias term.
            #We do not include 1 in the network initialization because this neuron "1" is not associated with a sensitivity value,
            #i.e it does not take part in backpropagating error. The bias neuron "1" hence is treated as separate entity as it's value is constant through all iterations
            self.ThreadsPerBlockClassify=(256,1)                                                            
            self.bpgX=(device_NodeOutputValues[self.layer_number].shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
            self.bpgY=(device_NodeOutputValues[self.layer_number].shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
            self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
            '''Kernel call'''
            copy[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_NodeOutputValues[self.layer_number], device_NodeOutputValues_withOne[self.layer_number]) 
            del self.bpgX,self.bpgY

        else:
            if(self.layer_number==max(device_arraySize.shape)-1):
                #Input is copied to output
                self.ThreadsPerBlockClassify=(16,16)                                                            
                self.bpgX=(device_OutputIdentityMatrix.shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
                self.bpgY=(device_OutputIdentityMatrix.shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
                self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
                '''Kernel call'''
                matmul[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_OutputIdentityMatrix, device_NodeInputValues[self.layer_number], device_NodeOutputValues[self.layer_number])
                del self.bpgX,self.bpgY 
            else:    
            #Input of present layer is passed through activation function to give the output of the present layer.
                self.ThreadsPerBlockClassify=(256,1)                                                           
                self.bpgX=(device_NodeOutputValues[self.layer_number].shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
                self.bpgY=(device_NodeOutputValues[self.layer_number].shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
                self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
                '''Kernel call'''
                Sigmoid[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_NodeInputValues[self.layer_number], device_NodeOutputValues[self.layer_number], device_networkParameters) 
                del self.bpgX,self.bpgY  
            
                #copy to new withOne array
                #We append 1 at the top of the output neuron array
                #This 1 takes care of the bias ter.
                #We do not include 1 in the network initialization because this neuron "1" is not associated with a sensitivity value,
                #i.e it does not take part in backpropagating error. The bias neuron "1" hence is treated as separate entity as it's value is constant through all iterations
                self.ThreadsPerBlockClassify=(256,1)                                                           
                self.bpgX=(device_NodeOutputValues[self.layer_number].shape[0]+self.ThreadsPerBlockClassify[0]-1)//self.ThreadsPerBlockClassify[0]
                self.bpgY=(device_NodeOutputValues[self.layer_number].shape[1]+self.ThreadsPerBlockClassify[1]-1)//self.ThreadsPerBlockClassify[1]
                self.BlocksPerGridClassify=(self.bpgX,self.bpgY)
                '''Kernel call'''
                copy[self.BlocksPerGridClassify,self.ThreadsPerBlockClassify](device_NodeOutputValues[self.layer_number], device_NodeOutputValues_withOne[self.layer_number]) 
                del self.bpgX,self.bpgY  

##################################################################### FEED FORWARD AND BACKPROPAGATION ################################################################
    
def feedforward(trainingSample,layer):
    '''
    Propagation of values across the layer
    layer is a list containing objects of Layer class for each layer in the network
    '''
    #print('Computing feedforward')
    for e in range(len(layer)):
        if e==0:
            layer[e].input(trainingSample)
            layer[e].output()
        else:
            layer[e].input(trainingSample)                                          
            layer[e].output()                                                   #dummy input, output does not use training sample if layer!=0     
            
                         
#device_outputData[sample],NNobject,device_error,device_errorPerIteration[sample]
def backpropagation(device_target, NNobject):
    '''
    The backpropgation algorithm is based on steps provided in Neural Networks and Learning Machines, Simon Haykin
    
    Implements gradient descent for convergence on some solution
    Caluclates error per node per iteration
    Stores the Kroencker delta values for each node which are then used for Weight updation
    Delta is also called as sensitivity of the node
    
    Input : old weights to modify
    Output : Weights modified after backpropagation
    '''
    #print('Computing backpropagation')
    #TRANSPOSE OF CUDA ARRAY CAUSES SO MUCH DELAY !!!
    
    #Calculating Delta Value for output node`
    ThreadsPerBlockClassify=(16,1)                                                           
    bpgX=(device_NodeOutputValues[-1].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
    bpgY=(device_NodeOutputValues[-1].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
    BlocksPerGridClassify=(bpgX,bpgY)
    DeltaOutput[BlocksPerGridClassify,ThreadsPerBlockClassify](device_target,device_NodeOutputValues[-1], device_NodeDeltaValues[-1], device_networkParameters) 
    #Output layer delta already computed, computing for last hidden backwards
    for i in range(len(NNobject.NodesByLayer)-2,0,-1):                                         
        temp=cuda.device_array(shape=(device_Weights[i].shape[1],1),dtype=np.float32)
        ThreadsPerBlockClassify=(16,16)                                                           
        bpgX=(device_NodeDeltaValues[i+1].shape[1]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
        bpgY=(device_Weights[i].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
        BlocksPerGridClassify=(bpgX,bpgY)
        #Multiplication of next layer's delta values with the weights between the two layers
        '''Kernel call'''
        transpose_matmul[BlocksPerGridClassify,ThreadsPerBlockClassify](device_NodeDeltaValues[i+1], device_Weights[i],temp)
        del bpgX,bpgY
        
        
        #Multiplying the above result with the derivative of activation function for corresponding node
        ThreadsPerBlockClassify=(16,1)                                                            
        bpgX=(device_NodeOutputValues[i].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
        bpgY=(device_NodeOutputValues[i].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
        BlocksPerGridClassify=(bpgX,bpgY)
        '''Kernel call'''
        DeltaHidden[BlocksPerGridClassify,ThreadsPerBlockClassify](device_NodeOutputValues[i], temp, device_NodeDeltaValues[i])
        del bpgX,bpgY
    
    #Calculating Weight Changes
    for i in range(len(NNobject.NodesByLayer)-1):
        ThreadsPerBlockClassify=(16,16)                                                           
        bpgX=(device_NodeDeltaValues[i+1].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
        bpgY=(device_NodeOutputValues_withOne[i].shape[0]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
        BlocksPerGridClassify=(bpgX,bpgY)
        '''Kernel call'''
        transpose2_matmul[BlocksPerGridClassify,ThreadsPerBlockClassify](device_NodeDeltaValues[i+1], device_NodeOutputValues_withOne[i],device_WeightChange[i])
        del bpgX,bpgY
       
    #Multiplying above with learning rate and adding the momentum of previous weight
    for i in range(len(NNobject.NodesByLayer)-1):
        ThreadsPerBlockClassify=(16,16)                                                            
        bpgX=(device_WeightChange[i].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
        bpgY=(device_WeightChange[i].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
        BlocksPerGridClassify=(bpgX,bpgY)
        '''Kernel call'''
        deltaWeight[BlocksPerGridClassify,ThreadsPerBlockClassify](device_WeightChange[i], device_networkParameters[2], device_PreviousWeightChange[i], device_networkParameters[3], device_networkParameters[4], device_Weights[i])
        del bpgX,bpgY
        
    #WeightUpdate
    for i in range(len(NNobject.NodesByLayer)-1):
        ThreadsPerBlockClassify=(16,16)                                                            
        bpgX=(device_Weights[i].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
        bpgY=(device_Weights[i].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
        BlocksPerGridClassify=(bpgX,bpgY)
        '''Kernel call'''
        weightUpdate[BlocksPerGridClassify,ThreadsPerBlockClassify](device_Weights[i], device_WeightChange[i])
     
         
    #Copying to previous Weight change
    for i in range(len(NNobject.NodesByLayer)-1):
        device_PreviousWeightChange[i]=device_WeightChange[i]
        
        
def error_function(device_target ,device_error ,device_errorPerIteration):
    #Calculate error for each node for that iteration
    ThreadsPerBlockClassify=(16,1)                                                           
    bpgX=(device_NodeOutputValues[-1].shape[0]+ThreadsPerBlockClassify[0]-1)//ThreadsPerBlockClassify[0]
    bpgY=(device_NodeOutputValues[-1].shape[1]+ThreadsPerBlockClassify[1]-1)//ThreadsPerBlockClassify[1]
    BlocksPerGridClassify=(bpgX,bpgY)
    '''Kernel call'''
    ErrorCalculate[BlocksPerGridClassify,ThreadsPerBlockClassify](device_error, device_NodeOutputValues[-1], device_target)
    
    #Total loss of the network for the iteration 
    ErrorReduce=cuda.Reduce(sum_reduce)  
    device_error=device_error.ravel()
    '''Kernel reduction call'''
    ErrorReduce(arr=device_error,res=device_errorPerIteration,init=0) 
            
def PreProcess(PathName,n_samples, trainValTestsplit):
    '''
    Pre processing the input dataset
    Input dataset is split into training, validation and testing datset
    Dataset is centred around origin with variance normalized to 1
    '''
    global inputDataTrain
    global outputDataTrain
    global inputDataValid
    global outputDataValid
    global inputDataTest
    global outputDataTest
    global inputScaler
    global outputScaler
    
    data=pd.read_csv(PathName)                                          #Take input from GUI
    data=np.array(data,dtype=np.float32)
    
    np.random.shuffle(data)
    data=data[0:n_samples,:]
    inputData=data[:,:-1].reshape([n_samples,data.shape[1]-1])
    outputData=data[:,-1].reshape([n_samples,1]) 
    
    #Splitting into Train, Test and Validation set
    inputDataTrain=data[:int(trainValTestsplit[0]*n_samples),:-1].reshape([int(trainValTestsplit[0]*n_samples),data.shape[1]-1])                                                                                              
    outputDataTrain=data[:int(trainValTestsplit[0]*n_samples),-1].reshape([int(trainValTestsplit[0]*n_samples),1])                                                                          #Separating target variable into output  
    #print(inputDataTrain)
    
    inputDataValid=data[int(trainValTestsplit[0]*n_samples):int(trainValTestsplit[1]*n_samples)+int(trainValTestsplit[0]*n_samples),:-1].reshape([int(trainValTestsplit[1]*n_samples),data.shape[1]-1])
    outputDataValid=data[int(trainValTestsplit[0]*n_samples):int(trainValTestsplit[1]*n_samples)+int(trainValTestsplit[0]*n_samples),-1].reshape([int(trainValTestsplit[1]*n_samples),1])
    #print(inputDataValid)
    
    inputDataTest=data[int(trainValTestsplit[1]*n_samples)+int(trainValTestsplit[0]*n_samples):,:-1].reshape([int(trainValTestsplit[2]*n_samples),data.shape[1]-1])
    outputDataTest=data[int(trainValTestsplit[1]*n_samples)+int(trainValTestsplit[0]*n_samples):,-1].reshape([int(trainValTestsplit[2]*n_samples),1])
    #print(inputDataTest)
    
    inputScaler=preprocessing.StandardScaler().fit(inputData)                                                                      #Converting to mean = 0 and variance = 1
    outputScaler=preprocessing.StandardScaler().fit(outputData) 
    
    #Normalizing the Train, Test and Validation set
    inputDataTrain=inputScaler.transform(inputDataTrain)
    outputDataTrain=outputScaler.transform(outputDataTrain) 
    inputDataValid=inputScaler.transform(inputDataValid)
    outputDataValid=outputScaler.transform(outputDataValid)
    inputDataTest=inputScaler.transform(inputDataTest)
    outputDataTest=outputScaler.transform(outputDataTest)
      
                                                                                                                                                            
    
def script(N_Samples,TrainTestValidation,NetArch,Hyperparams,MaxEpochs,SaveLocation,DatasetDirectory):
    '''
    Takes inputs from ANNGui
    Assigns it to local variables which are used to initialize variables, Neural Network, Layers, Feedforward, Backpropagation
    '''
    
    global ErrorPerIterationTrain
    global ErrorPerIterationValid
    global ErrorPerIterationTest
    global ErrorPerEpochTrain
    global ErrorPerEpochValid
    global ErrorPerEpochTest
    global NumberOfSamplesTrain
    global EpochMax
    global ErrorPerIteration
    
    
    
    PreProcess(DatasetDirectory, N_Samples, TrainTestValidation)
    
    NumberOfSamplesTrain=inputDataTrain.shape[0]
    NumberOfSamplesValid=inputDataValid.shape[0]
    NumberOfSamplesTest=inputDataTest.shape[0]

    
    x=NetArch  
    y=Hyperparams
    EpochMax=MaxEpochs
    print('Starting initialization')
    NNobject=NeuralNetwork(x,y)
    
    #Transferring CPU variables to GPU memory (cuda.to_device()), Allocating GPU memory (cuda.device_array())
    #Separate memory for Training, Testing and Validation variables
    device_errorTrain=cuda.device_array(shape=(NNobject.NodesByLayer[-1],1),dtype=np.float32)
    device_errorValid=cuda.device_array(shape=(NNobject.NodesByLayer[-1],1),dtype=np.float32)
    device_errorTest=cuda.device_array(shape=(NNobject.NodesByLayer[-1],1),dtype=np.float32)
    device_inputDataTrain=cuda.to_device(np.matrix(inputDataTrain,dtype=np.float32))
    device_outputDataTrain=cuda.to_device(np.matrix(outputDataTrain,dtype=np.float32))
    device_inputDataValid=cuda.to_device(np.matrix(inputDataValid,dtype=np.float32))
    device_outputDataValid=cuda.to_device(np.matrix(outputDataValid,dtype=np.float32))
    device_inputDataTest=cuda.to_device(np.matrix(inputDataTest,dtype=np.float32))
    device_outputDataTest=cuda.to_device(np.matrix(outputDataTest,dtype=np.float32))
    device_errorPerIterationTrain=cuda.device_array(shape=(NumberOfSamplesTrain,1),dtype=np.float32)
    device_errorPerIterationValid=cuda.device_array(shape=(NumberOfSamplesValid,1),dtype=np.float32)
    device_errorPerIterationTest=cuda.device_array(shape=(NumberOfSamplesTest,1),dtype=np.float32)
    
    epoch=0
    iteration=0
    layer=[]
    #creates objects for each layer, identifies them with number given by t
    for t in range(len(NNobject.NodesByLayer)):                                     
        layer.append(Layers(t))
    print('Layers initialized')
    
    start=time.time()
    while(epoch<EpochMax):
        for sample in range(NumberOfSamplesTrain):
            #Picking each sample from the entire dataset
            inputSampleTrain=device_inputDataTrain[sample,:]
            outputSampleTrain=device_outputDataTrain[sample]
            inputSampleTrain=inputSampleTrain.reshape([NNobject.NodesByLayer[0],1])
            outputSampleTrain=outputSampleTrain.reshape([NNobject.NodesByLayer[-1],1])
            
            feedforward(inputSampleTrain,layer)
            error_function(outputSampleTrain,device_errorTrain,device_errorPerIterationTrain[sample])
            
            #Implementing feedfoward for validation set too, to plot validation curve alongside training curve
            if(sample<NumberOfSamplesValid):
                inputSampleValid=device_inputDataValid[sample,:]
                outputSampleValid=device_outputDataValid[sample]
                inputSampleValid=inputSampleValid.reshape([NNobject.NodesByLayer[0],1])
                outputSampleValid=outputSampleValid.reshape([NNobject.NodesByLayer[-1],1])
                feedforward(inputSampleValid,layer)
                error_function(outputSampleValid,device_errorValid,device_errorPerIterationValid[sample]) 
            #Backpropagation is done only for the training set    
            backpropagation(outputSampleTrain,NNobject)
            iteration+=1
        
        #Calculating Error per epoch for Training and Validation, used to plot the Error vs Epoch curve, Early stopping       
        ErrorPerIterationTrain=device_errorPerIterationTrain.copy_to_host()
        ErrorPerIterationValid=device_errorPerIterationValid.copy_to_host()
        tempError=0
        for e in ErrorPerIterationTrain:
            tempError+=e[0]
        tempError/=NumberOfSamplesTrain
        ErrorPerEpochTrain.append(tempError)
        tempError=0
        for e in ErrorPerIterationValid:
            tempError+=e[0]
        tempError/=NumberOfSamplesValid
        ErrorPerEpochValid.append(tempError)
        epoch+=1 
        print(epoch) 
    
    end=time.time()    
    #Error Computation for Test Data 
    for sample in range(NumberOfSamplesTest):
        inputSampleTest=device_inputDataTest[sample,:]
        outputSampleTest=device_outputDataTest[sample]
        inputSampleTest=inputSampleTest.reshape([NNobject.NodesByLayer[0],1])
        outputSampleTest=outputSampleTest.reshape([NNobject.NodesByLayer[-1],1])
        feedforward(inputSampleTest,layer) 
        error_function(outputSampleTest,device_errorTest,device_errorPerIterationTest[sample])  
    ErrorPerIterationTest=device_errorPerIterationTest.copy_to_host() 
    tempError=0
    for e in ErrorPerIterationTest:
        tempError+=e[0]
    tempError/=NumberOfSamplesTest
    ErrorPerEpochTest=tempError
    print('Error on Test Data Set: '+str(ErrorPerEpochTest))
    
    
    #Saving the Weights
    FinalWeights=[0]*(len(NetArch)-1)
    for i in range(len(NetArch)-1):
        FinalWeights[i]=device_Weights[i].copy_to_host()
        
    pickle_T=open(SaveLocation+'FinalWeight.pkl','wb')
    pickle.dump(FinalWeights,pickle_T)
    pickle_T.close()
    
    
    plt.figure() 
    plt.ylabel("Error")
    plt.xlabel("Number of Epochs")
    a=plt.plot(ErrorPerEpochTrain,'r')
    b=plt.plot(ErrorPerEpochValid,'b') 
    plt.legend((a[0],b[0]),('Training Error','Validation Error'))
    
    return(ErrorPerEpochTest,end-start)
#script()
 

    