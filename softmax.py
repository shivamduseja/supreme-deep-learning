import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as functional


class Softmax(nn.Module):
    def __init__(self, im_size, n_classes):
        '''
        Create components of a softmax classifier and initialize their weights.

        Arguments:
            im_size (tuple): A tuple of ints with (channels, height, width)
            n_classes (int): Number of classes to score
        '''
        super(Softmax, self).__init__()
        #############################################################################
        # TODO: Initialize anything you need for the forward pass
        #############################################################################
        #pass
        
        self.n_classes=n_classes
        D=im_size[0]*im_size[1]*im_size[2]
        self.layer1=nn.Linear(D,self.n_classes)
        self.layer2=nn.Softmax(dim=1)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################

    def forward(self, images):
        '''
        Take a batch of images and run them through the classifier to
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
        # TODO: Implement the forward pass. This should take very few lines of code.
        #############################################################################
        # pass
        
        D1=images.view(images.shape[0],-1)
        y=self.layer1(D1)
        scores=self.layer2(y)
        
        #############################################################################
        #                             END OF YOUR CODE                              #
        #############################################################################
        return scores

