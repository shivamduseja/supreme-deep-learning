import torchvision.models as models

import torch
import torch.nn as nn
import torch.nn.functional as functional
from torchsummary import summary


class Resnet(nn.Module):
    def __init__(self, kind, part, my_model):
        
        super(Resnet, self).__init__()
 
        res_model = models.resnet18(pretrained = True)
        
        #self.res_layer = nn.Sequential(*list(res_model.children())[:-2])
        self.res_layer = res_model
        self.kind = kind
        self.part = part
        self.my_model = my_model
        
        num_ftrs = res_model.fc.in_features
        out_ftrs = res_model.fc.out_features
        
        for p in self.res_layer.parameters():
            p.requires_grad = False
        
        if (self.kind == 'softmax'):
            self.res_layer.fc = nn.Linear(num_ftrs,10)
            
        if (self.kind == 'twolayernn'):
            self.res_layer.fc = nn.Sequential(nn.Linear(num_ftrs, 1000),
                                             nn.Linear(1000, 10),
                                             nn.Softmax())
        if (self.kind == 'convnet'):
            del self.res_layer.avgpool
            self.res_layer.fc = nn.Sequential(
                    nn.Conv2D(out_ftrs,120,1,1),
                    nn.Dropout2D(0.5), 
                    nn.ReLU(inplace= TRUE), 
                    nn.MaxPool2D(3,1), nn.Linear(120, 10))
            
        if part == 1:
            pass
            #self.res_layer.fc.requires_grad = False         #self.my_model.requires_grad = False
        if part == 2:
            #With updatable parameters
            self.res_layer.fc.requires_grad = True
        if part == 3:
            #back prop to entire resnet layer
            for p in self.res_layer.parameters():
                p.requires_grad = True
            self.res_layer.fc.requires_grad = True          #self.my_model.requires_grad = True
          
    def forward(self, images):
        scores = None 
        
        x = images.view(images.shape[0],-1)
        x = self.res_layer(images)
        
        scores = x
        return scores

# Practice code : keeping it for future need!!   
#class Identity(nn.Module):
#    def __init__(self):
#        super().__init__()
    
#    def forward(self, x):
#        return x

#c = 0
            #for child in self.res_layer.children():
            #    print(child)
            #    c = c +1
            #    if c <= 7:
            #        for param in child.parameters():
            #            param.requires_grad = False
            #self.res_layer.fc.requires_grad = True          #self.my_model.requires_grad = True

#print("printing size from res layer")
        #print(x.shape)
        
        #images = images.reshape(images.shape[0],3, 32, 32)
        #print("after reshape")
        #print(images.shape)
        
        #x = self.my_model(x)
        #print("printing shape after my model")
        #print(x.shape)

#self.res_layer = nn.Sequential(*list(self.res_layer.children())[:-1])   