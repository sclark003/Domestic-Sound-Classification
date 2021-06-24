import torch
from torch import nn
import torch.nn.functional as F


class AudioClassifier (nn.Module):

    def __init__(self):
        super().__init__()

        self.att_block = AttBlock(2048, 10, activation='sigmoid')
        self.interpolate_ratio = 32
     
        self.bn0 = nn.BatchNorm2d(64)

        self.conv_block1 = ConvBlock(in_channels=1, out_channels=64)
        self.conv_block2 = ConvBlock(in_channels=64, out_channels=128)
        self.conv_block3 = ConvBlock(in_channels=128, out_channels=256)
        self.conv_block4 = ConvBlock(in_channels=256, out_channels=512)
        self.conv_block5 = ConvBlock(in_channels=512, out_channels=1024)
        self.conv_block6 = ConvBlock(in_channels=1024, out_channels=2048)

        self.fc1 = nn.Linear(2048, 2048, bias=True)
 
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
    
    
    def interpolate(self, x: torch.Tensor, ratio: int):
        (batch_size, time_steps, classes_num) = x.shape
        upsampled = x[:, :, None, :].repeat(1, 1, ratio, 1)
        upsampled = upsampled.reshape(batch_size, time_steps * ratio, classes_num)
        return upsampled
    
    
    def pad_framewise_output(self,framewise_output: torch.Tensor, frames_num: int):
        pad = framewise_output[:, -1:, :].repeat(1, frames_num - framewise_output.shape[1], 1)
        output = torch.cat((framewise_output, pad), dim=1)
        return output
    

    def forward(self, x): # input = [16, 1, 64, 2579]
        # Run the convolutional blocks
        frames_num = x.shape[3] # [16, 1, 2579, 64], frames_num = 2579
        x = x.transpose(2, 3)
        x = self.cnn_feature_extractor(x) # [16, 2048, 80, 2]
          
        #x = self.conv(x)        # [16, 64, 4, 162]
  
        # Adaptive pool and flatten for input to linear layer
        #x = self.ap(x)
        #x = x.view(x.shape[0], -1)

        # Linear layer
        #x = self.lin(x)      

        # Aggregate in frequency axis
        x = torch.mean(x, dim=3)  # [16, 2048, 80]
        
        x = F.dropout(x, p=0.5, training=self.training)  # [16, 2048, 80]
        x = x.transpose(1, 2)                            # [16, 80, 2048]
        x = F.relu_(self.fc1(x))                         # [16, 80, 2048]
        x = x.transpose(1, 2)                            # [16, 2048, 80]  
        x = F.dropout(x, p=0.5, training=self.training)  # [16, 2048, 80]  

        (clipwise_output, norm_att, segmentwise_output) = self.att_block(x)
        # clipwise_output.shape      =   [16, 10]
        # norm_att.shape             =   [16, 10, 80]
        # segmentwise_output.shape   =   [16, 10, 80]
        segmentwise_output = segmentwise_output.transpose(1, 2)   # [16, 80, 10]  

        # Get framewise output
        framewise_output = self.interpolate(segmentwise_output,self.interpolate_ratio)   # [16, 2560, 10]                             
        framewise_output = self.pad_framewise_output(framewise_output, frames_num)       # [16, 2579, 10]                              

        output_dict = {'framewise_output': framewise_output,'clipwise_output': clipwise_output}
        
        return output_dict


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


class AttBlock(nn.Module):
    def __init__(self,
                 in_features: int,
                 out_features: int,
                 activation="linear",
                 temperature=1.0):
        super().__init__()

        self.activation = activation
        #self.temperature = temperature
        self.att = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)
        self.cla = nn.Conv1d(
            in_channels=in_features,
            out_channels=out_features,
            kernel_size=1,
            stride=1,
            padding=0,
            bias=True)

        self.bn_att = nn.BatchNorm1d(out_features)
        self.init_weights()

    def init_layer(self,layer):
        nn.init.xavier_uniform_(layer.weight)
    
        if hasattr(layer, "bias"):
            if layer.bias is not None:
                layer.bias.data.fill_(0.) 
    
    def init_bn(self,bn):
        bn.bias.data.fill_(0.)
        bn.weight.data.fill_(1.0)
        
    
    def init_weights(self):
        self.init_layer(self.att)
        self.init_layer(self.cla)
        self.init_bn(self.bn_att)

    def forward(self, x):
        # x: (n_samples, n_in, n_time)
        norm_att = torch.softmax(torch.clamp(self.att(x), -10, 10), dim=-1)    # aggregate the classification result for segment
        cla = self.nonlinear_transform(self.cla(x))                            # calculates segment wise classification result
        x = torch.sum(norm_att * cla, dim=2)                                   # attention aggregation is performed to get clip wise prediction
        return x, norm_att, cla

    def nonlinear_transform(self, x):
        if self.activation == 'linear':
            return x
        elif self.activation == 'sigmoid':
            return torch.sigmoid(x)
        
        
class PANNsLoss(nn.Module):
    def __init__(self):
        super().__init__()

        self.bce = nn.BCELoss()

    def forward(self, input, target):
        input_ = input["clipwise_output"]
        input_ = torch.where(torch.isnan(input_),
                             torch.zeros_like(input_),
                             input_)
        input_ = torch.where(torch.isinf(input_),
                             torch.zeros_like(input_),
                             input_)
        target = target.float()

        return self.bce(input_, target)
    
    
    
    
        # self.fc1 = nn.Linear(64,64, bias=True)       

        # # First Convolution Block with Relu and Batch Norm. Use Kaiming Initialization
        # self.conv1 = nn.Conv2d(2, 8, kernel_size=(5, 5), stride=(2, 2), padding=(2, 2))
        # self.relu1 = nn.ReLU()
        # self.bn1 = nn.BatchNorm2d(8)
        # init.kaiming_normal_(self.conv1.weight, a=0.1)
        # self.conv1.bias.data.zero_()
        # conv_layers += [self.conv1, self.relu1, self.bn1]

        # # Second Convolution Block
        # self.conv2 = nn.Conv2d(8, 16, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.relu2 = nn.ReLU()
        # self.bn2 = nn.BatchNorm2d(16)
        # init.kaiming_normal_(self.conv2.weight, a=0.1)
        # self.conv2.bias.data.zero_()
        # conv_layers += [self.conv2, self.relu2, self.bn2]

        # # Second Convolution Block
        # self.conv3 = nn.Conv2d(16, 32, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.relu3 = nn.ReLU()
        # self.bn3 = nn.BatchNorm2d(32)
        # init.kaiming_normal_(self.conv3.weight, a=0.1)
        # self.conv3.bias.data.zero_()
        # conv_layers += [self.conv3, self.relu3, self.bn3]

        # # Second Convolution Block
        # self.conv4 = nn.Conv2d(32, 64, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1))
        # self.relu4 = nn.ReLU()
        # self.bn4 = nn.BatchNorm2d(64)
        # init.kaiming_normal_(self.conv4.weight, a=0.1)
        # self.conv4.bias.data.zero_()
        # conv_layers += [self.conv4, self.relu4, self.bn4]

        # # Linear Classifier
        # self.ap = nn.AdaptiveAvgPool2d(output_size=1)
        # self.lin = nn.Linear(in_features=64, out_features=10)

        # # Wrap the Convolutional Blocks
        # self.conv = nn.Sequential(*conv_layers)