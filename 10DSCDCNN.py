# -*- coding: utf-8 -*-
"""
Created on Sat Apr  7 14:29:00 2018

@author: Rehan
"""
import torch
import torch.nn.functional as F
from torch.autograd import Variable
import numpy as np
import matplotlib.pyplot as plt
import cv2
from time import time
import pdb

class DCNN(torch.nn.Module):
    def __init__(self):
        """
        In the constructor we instantiate Convolution 
        and DeConvolution modules and assign them as member variables.
        """
        super(DCNN, self).__init__()
        self.conv1 = torch.nn.Conv2d(in_channels=3,out_channels=64, kernel_size=3, padding=1, stride=1)
        self.conv2 = torch.nn.Conv2d(in_channels=64,out_channels=64, kernel_size=3, padding=1, stride=2)
        self.conv3 = torch.nn.Conv2d(in_channels=64,out_channels=128, kernel_size=3, padding=1, stride=2)
        self.conv4 = torch.nn.Conv2d(in_channels=128,out_channels=256, kernel_size=3, padding=1, stride=2)
        self.conv5 = torch.nn.Conv2d(in_channels=256,out_channels=512, kernel_size=3, padding=1, stride=2)

        self.deconv1 = torch.nn.ConvTranspose2d(in_channels=512, out_channels=256, kernel_size= 3, padding=1, stride=2, output_padding=1)
        self.deconv2 = torch.nn.ConvTranspose2d(in_channels=256, out_channels=128, kernel_size= 3, padding=1, stride=2, output_padding=1)
        self.deconv3 = torch.nn.ConvTranspose2d(in_channels=128, out_channels=64, kernel_size= 3, padding=1, stride=2, output_padding=1)
        self.deconv4 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=64, kernel_size= 3, padding=1, stride=2, output_padding=1)
        self.deconv5 = torch.nn.ConvTranspose2d(in_channels=64, out_channels=3, kernel_size= 3, padding=1, stride=1)
        
        self.delta1 = 0.1
        self.delta2 = 0.2
        self.delta3 = 0.3
        self.delta4 = 0.4
        
    def forward(self, x):
        """
        In the forward function we accept a Variable of input data and we must return
        a Variable of output data. We can use Modules defined in the constructor as
        well as arbitrary operators on Variables. In forward we connect the 10 layers 
        in sequence as defined in the research paper.
        """
        layer1 = F.relu(self.conv1(x)) # First Convolution layer 
        # previous layer scaled with delta and feeding to next layer.
        # See the equation of X_i in the paper. Then applying Relu.
        layer2 = F.relu(self.conv2(torch.mul(layer1,self.delta1))) 
        layer3 = F.relu(self.conv3(torch.mul(layer2,self.delta2)))
        layer4 = F.relu(self.conv4(torch.mul(layer3,self.delta3)))
        
        layer5 = F.relu(self.conv5(torch.mul(layer4,self.delta4)))

        layer6 = F.relu(self.deconv1(layer5))
        # previous layer scaled with "1-delta" and adding the symmetric layer output 
        # then feeding to next layer. See the equation of X'_i in the paper.
        layer7 = F.relu(self.deconv2(torch.mul(layer4,1-self.delta1) + layer6))
        layer8 = F.relu(self.deconv3(torch.mul(layer3,1-self.delta2) + layer7))
        layer9 = F.relu(self.deconv4(torch.mul(layer2,1-self.delta3) + layer8))
        output = F.relu(self.deconv5(torch.mul(layer1,1-self.delta4) + layer9))
        return output

class ConvertDat():
    def __int__(self):
        pass

    def unpickle(self, file):
        import pickle
        with open(file, 'rb') as fo:
            dict = pickle.load(fo, encoding='bytes')
        return dict

    def GetData(self, dataDict, verbose=False):
        dataSet = dataDict[list(dataDict.keys())[2]]    
        # Create the images out of vector shaped data set. 
        Images = np.ndarray((10000,3,64,64),dtype=np.uint8)
        for i,d in enumerate(dataSet):
             k = d.reshape((3,32,32))
             Images[i,0,:,:] = cv2.resize(k[0,:,:], (64,64))
             Images[i,1,:,:] = cv2.resize(k[1,:,:], (64,64))
             Images[i,2,:,:] = cv2.resize(k[2,:,:], (64,64))        
        # Display one random example image
        if verbose:
            plt.figure(figsize=(3,3))
            plt.imshow(np.transpose(Images[np.random.randint(0,10000),:,:,:],(1,2,0)),interpolation='bilinear')
            plt.title('Any random image from data set...')
        return Images

class TrainTest():

    def __int__(self):
        pass
    
    def Train(self, epochs, X, Y, verbose=True):
        # Construct our model by instantiating the class defined above
        self.model = DCNN()
        # Construct our loss function and an Optimizer. The call to model.parameters()
        # in the SGD constructor will contain the learnable parameters of the ten
        # modules which are members of the model.
        self.criterion = torch.nn.MSELoss()
        self.optimizer = torch.optim.SGD(self.model.parameters(), lr=1e-5)
        self.model.cuda()
        self.error = np.ndarray((epochs,))
        
        for t in range(epochs):
            # Forward pass: Compute predicted Y_pred by passing Xs to the model
            Y_pred = self.model(X)
    
            # Compute and print loss
            loss = self.criterion(Y_pred, Y)
            self.error[t] = loss.data[0]
            print(t, loss.data[0])
    
            # Zero gradients 
            self.optimizer.zero_grad()
            # Perform a backward pass (back propagation)
            loss.backward()
            # Update the weights.
            self.optimizer.step()
    
        plt.figure(); plt.plot(self.error)
        plt.xlabel('epoch'); plt.ylabel('MSE'); plt.title('MSE Error plot')
    
        if verbose:
            plt.figure(figsize=(2,2))
            plt.imshow(np.transpose(np.uint8(Y.data[0,:,:,:]), (1,2,0)), interpolation='bilinear')
            plt.title('Original Non-noisey Image')
    
            plt.figure(figsize=(2,2))
            plt.imshow(np.transpose(np.uint8(Y_pred.data[0,:,:,:]), (1,2,0)), interpolation='bilinear')
            plt.title('Reconstructed image after training')

    def Test(self, Data, Y):
        Y_pred = self.model(Data)
        loss = self.criterion(Y_pred, Y)
        print('Test MSE is ', loss.data[0])

if __name__ == '__main__':
    tic = time()
    dataSetPath = 'D:/Projects_Lectures/freelancing_projects/Daniel_pytorch_DeconvCNN/cifar-10-python/cifar-10-batches-py/data_batch_1'    
    verbose = True
    epochs = 2000
    
    # unpickle the cifar10 data set.
    CD = ConvertDat()
    dataDict = CD.unpickle(dataSetPath)
    # Get the Image data from Dictionary
    Images = CD.GetData(dataDict, verbose)
    
    # input Image/data dimesnion should be (batchsize, channels, width, height)
    n_examples = 150 # number of examples to train the Network. 
    X = Variable(torch.cuda.FloatTensor(Images[0:n_examples,:,:,:]))
    X_noisy = Variable(torch.cuda.FloatTensor(Images[0:n_examples,:,:,:] + np.random.randn(n_examples,3,64,64) + 1))

#    if verbose:
#        plt.figure(figsize=(2,2))
#        plt.imshow(np.transpose(np.uint8(X[0,:,:,:]), (1,2,0)), interpolation='bilinear')
#        plt.title('Noisy Image')

    TT = TrainTest()
    TT.Train(epochs, X_noisy, X, verbose)
#    TT.Test(Xtest_noisy, Xtest)
    toc = time()
    print('Total Time of Execution %.2f min' %((toc-tic)/60.0))