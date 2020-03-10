import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as functional


class CNN(nn.Module):
    def __init__(self, im_size, hidden_dim, kernel_size, n_classes):
        '''
        Create components of a CNN classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            hidden_dim (int): Number of hidden activations to use
            kernel_size (int): Width and height of (square) convolution filters
            n_classes (int): Number of classes to score
        '''
        super(CNN, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        pass
        self.conv1=nn.Conv2d(im_size[0],8,kernel_size)
        #self.relu=nn.Relu()
        self.pool=nn.MaxPool2d(kernel_size=kernel_size)
        poolstride=1
        # Lout=floor((Lin+2∗padding−dilation∗(kernel_size−1)−1)/stride+1)
        
        op_size=np.floor(im_size[1]-(kernel_size-1)-1)/(poolstride+1)
        op_size=op_size.astype(np.int64)
        #print('opsize',op_size)
        self.l1=nn.Linear(8*op_size*op_size,hidden_dim)
        self.l2=nn.Linear(hidden_dim,n_classes)
        self.layer2=nn.Softmax(dim=1)
               
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the CNN to
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
        # TODO: Implement the forward pass. This should take few lines of code.
        #############################################################################
        pass
    
        img_data=images.view(images.shape[0],-1)
       
        c1=(self.conv1(images)).clamp(min=0)
        c2=self.pool(c1)
        #print('c1',c2.shape,c1.shape)
        
        l1=self.l1(c2.view(c2.shape[0],-1))
        #print('l1',l1.shape)
        
        scores=(self.l2(l1))
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

