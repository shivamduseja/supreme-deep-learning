import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class MyModel(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Extra credit model

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(MyModel, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        pass
        outchannel1=32
        outchannel2=64
        outchannel3=128
        outchannel4=256
        #poolstride=kernel_size-1
        
        self.conv_layer = nn.Sequential(
        # Conv Layer 1
            nn.Conv2d(3,outchannel1,kernel_size, padding=1),
            nn.BatchNorm2d(outchannel1),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel1,outchannel2, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),nn.Dropout2d(p=0.01))
        
        # Conv Layer 2
        self.conv_layer1=nn.Sequential(
            nn.Conv2d(outchannel2, outchannel3, kernel_size, padding=1),
            nn.BatchNorm2d(outchannel3),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel3, outchannel3, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2),
            nn.Dropout2d(p=0.04))
        
        # Conv Layer 3
        self.conv_layer2=nn.Sequential(
            nn.Conv2d(outchannel3, outchannel4, kernel_size, padding=1),
            nn.BatchNorm2d(outchannel4),
            nn.ReLU(inplace=True),
            nn.Conv2d(outchannel4, outchannel4, kernel_size, padding=1),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(2,2))
        
        #self.l1=nn.Dropout2d(0.1),
        #self.l2=nn.Linear(outchannel4*16,hidden_dim),
        #self.l3=nn.ReLU(inplace=True),
            #nn.Linear(hidden_dim[0],hidden_dim[1]),
        #self.l4=nn.Linear(hidden_dim,n_classes)
        
        self.linear_layer=nn.Sequential(nn.Dropout2d(0.1),nn.Linear(outchannel4*16,hidden_dim),
        nn.ReLU(inplace=True),nn.Linear(hidden_dim,n_classes))
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the model to
        produce a score for each class.

        Arguments:
            images (Variable): A tensor of size (N, C, H, W) where
                N is the batch size
                C is the number of channels
                H is the image height
                W is the image width

        Returns:
            A torch Variable of size (N, n_classes) specifying the score
            for each example and category.
        '''
        scores = None
        #############################################################################
        # TODO: Implement the forward pass.
        #############################################################################
        pass
        
        c1=self.conv_layer(images)
        c2=self.conv_layer1(c1)
        c3=self.conv_layer2(c2)
        
        img_data=c3.view(c3.shape[0],-1)
        #print(img_data.shape)
        ##l1=self.l1(img_data)
        #l2=self.l2(img_data)
        #l3=self.l3(l2)
        scores=self.linear_layer(img_data)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores
