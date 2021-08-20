"""
CNN-14 and ResNet models for classification
"""


import torch.nn.functional as F
from torch.nn import init
import torch.nn as nn
import torch


"""
----------------------------
ResNet Classification Model
----------------------------
Add output layer to ResNet model
"""
class Net(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.model = model               
        self.fc1 = nn.Linear(1000, 7)   
    
    def forward(self, x): # input = [30, 3, 64, 128]
        x = self.model(x) # load pytorch model
        x = self.fc1(x)   # output 7 class predictions
        return x


"""
----------------------------
CNN-14 Classification Model
----------------------------
Adapted from: https://github.com/qiuqiangkong/audioset_tagging_cnn/blob/master/pytorch/models.py
Qiuqiang Kong, Yin Cao, Turab Iqbal, Yuxuan Wang, Wenwu Wang, and Mark D. Plumbley. 
"Panns: Large-scale pretrained audio neural networks for audio pattern recognition." 
IEEE/ACM Transactions on Audio, Speech, and Language Processing 28 (2020): 2880-2894.
"""
class AudioClassifier (nn.Module):

    def __init__(self):
        super().__init__()

        #self.activation == 'sigmoid'
     
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=3, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
        
        self.att = nn.Conv1d(in_channels=2048,out_channels=7,kernel_size=1,stride=1,padding=0,bias=True)
        self.cla = nn.Conv1d(in_channels=2048,out_channels=7,kernel_size=1,stride=1,padding=0,bias=True)
        

 
    def cnn_feature_extractor(self, x):
        x = self.conv_block1(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block2(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block3(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block4(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block5(x, pool_size=(2, 2), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        x = self.conv_block6(x, pool_size=(1, 1), pool_type='avg')
        x = F.dropout(x, p=0.2, training=self.training)
        return x
    
    

    def forward(self, x):                        # input = [30, 3, 128, 87] (batch size, channels, mels, time frames)   
        # Run the convolutional blocks
        x = x.transpose(2, 3)                            # [30, 3, 87, 128]
        x = self.cnn_feature_extractor(x)                # [30, 2048, 2, 4]
        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)                         # [30, 2048, 2]
        x = F.dropout(x, p=0.5, training=self.training)  # [30, 2048, 2]
        x = x.transpose(1, 2)                            # [30, 2, 2048]
        x = F.relu_(self.fc1(x))                         # [30, 2, 2048]
        x = x.transpose(1, 2)                            # [30, 2048, 2] 
        x = F.dropout(x, p=0.5, training=self.training)  # [30, 2048, 2] (n_samples, n_in, n_time)
        
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)    # aggregate the classification result for segment
        cla = torch.sigmoid(self.cla(x))                                       # calculates segment wise classification result
        x = torch.sum(norm_att * cla, dim=2)                                   # attention aggregation is performed to get clip wise prediction  
        
        return x


class ConvBlock(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()

        self.conv1 = nn.Conv2d(
            in_channels=in_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.conv2 = nn.Conv2d(
            in_channels=out_channels,
            out_channels=out_channels,
            kernel_size=(3, 3),
            stride=(1, 1),
            padding=(1, 1),
            bias=False)

        self.bn1 = nn.BatchNorm2d(out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)

        self.init_weight()

    def init_layer(self,layer):
        nn.init.xavier_uniform_(layer.weight)    

    def init_bn(self,bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)    

    
    def init_weight(self):
        self.init_layer(self.conv1)
        self.init_layer(self.conv2)
        self.init_bn(self.bn1)
        self.init_bn(self.bn2)

    def forward(self, input, pool_size=(2, 2), pool_type='avg'):

        x = input
        x = F.relu_(self.bn1(self.conv1(x)))
        x = F.relu_(self.bn2(self.conv2(x)))
        if pool_type == 'max':
            x = F.max_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg':
            x = F.avg_pool2d(x, kernel_size=pool_size)
        elif pool_type == 'avg+max':
            x1 = F.avg_pool2d(x, kernel_size=pool_size)
            x2 = F.max_pool2d(x, kernel_size=pool_size)
            x = x1 + x2
        else:
            raise Exception('Incorrect argument!')

        return x
