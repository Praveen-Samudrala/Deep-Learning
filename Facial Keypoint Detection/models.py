## TODO: define the convolutional neural network architecture

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.nn.init as I


class Net(nn.Module):

    def __init__(self):
        super(Net, self).__init__()
        
        self.conv1 = nn.Conv2d(1, 32, 5)
        # Image size is [1,224,224] - output size =(W-F)/s +1 --> (224-5)/1 +1 = 220
        # After maxpool with 2x2 window, size -> [32,110,110]
        self.pool = nn.MaxPool2d(2,2)
        
        self.conv2 = nn.Conv2d(32, 64, 5) # output = (110-5)/1 +1 = 106. After max pooling, [64,53,53]
        self.conv3 = nn.Conv2d(64, 128, 3) # output = (53-3)/1 +1 = 51, after max pooling, [128,25,25]
        self.conv4 = nn.Conv2d(128, 256, 3) #output = (25-3)/1 +1 = 23, after max pooling, [256,11,11]
        self.conv5 = nn.Conv2d(256, 512, 3) # output = (11-3)/1 +1 = 9, after max pooling, [512,4,4] - each feature map [1,4,4]

        #self.batchnorm32 = nn.BatchNorm2d(32)

        self.dense1 = nn.Linear(512*4*4, 3000)
        self.dense2 = nn.Linear(3000, 800)
        self.dense3 = nn.Linear(800, 136)
        
        self.drop1 = nn.Dropout(p=0.1)
        self.drop2 = nn.Dropout(p=0.2)
        self.drop3 = nn.Dropout(p=0.3)
        self.drop4 = nn.Dropout(p=0.4)
        self.drop5 = nn.Dropout(p=0.4)
        self.drop6 = nn.Dropout(p=0.4)
        self.drop7 = nn.Dropout(p=0.4)


    def forward(self, x):

        x = self.pool(F.relu(self.conv1(x)))
        x = self.drop1(x)
        
        x = self.pool(F.relu(self.conv2(x)))
        x = self.drop2(x)
        
        x = self.pool(F.relu(self.conv3(x)))
        x = self.drop3(x)
        
        x = self.pool(F.relu(self.conv4(x)))
        x = self.drop4(x)
        
        x = self.pool(F.relu(self.conv5(x)))
        x = self.drop5(x)
        
        
        x = x.view(x.size(0), -1)
        
        
        x = F.relu(self.dense1(x))
        x = self.drop6(x)
        
        x = F.relu(self.dense2(x))
        x = self.drop7(x)
        
        x = self.dense3(x)
        # Not putting a softmax because this is regression, not classification
        return x
