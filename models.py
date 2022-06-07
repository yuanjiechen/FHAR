import copy

from pydoc import locate
import torch
import torch.nn as nn
import torch.nn.functional as F


class ConvUnit(nn.Module):

    def __init__(self, in_channels, out_channels, kernel_size, stride):
        super(ConvUnit, self).__init__()
        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=stride)
        self.bn   = nn.BatchNorm2d(out_channels)

    def forward(self, x):
        return F.relu(self.bn(self.conv(x)))

# class MM_EAT(nn.module):
#     def __init__(self, in_shape:list, out_shape:list, n_classes, kernel_size=[7, 5, 5], stride=[2, 2, 1], pool_kernal=[2, 2, 2], pool_stride=[2, 2, 2]) -> None:
#         super().__init__()


#     def forward(self):
#         pass


class CNN_LN(nn.Module):

    def __init__(self, n_classes, in_shape=[150, 96, 128, 3584, 512, 128], out_shape=[96, 128, 512, 512, 128], kernel_size=[3, 3, 3], stride=[2, 1, 1], pool_kernal=[2, 2, 2], pool_stride=[2, 1, 1]) -> None:
        super().__init__()

        self.pool_kernal = pool_kernal
        self.pool_stride = pool_stride

        self.input_shape = in_shape
        self.output_shape = out_shape

        self.conv1 = ConvUnit(in_channels=self.input_shape[0], out_channels=self.output_shape[0], kernel_size=kernel_size[0], stride=stride[0])
        self.conv2 = ConvUnit(in_channels=self.input_shape[1], out_channels=self.output_shape[1], kernel_size=kernel_size[1], stride=stride[1])
        #self.conv3 = ConvUnit(in_channels=self.input_shape[2], out_channels=self.output_shape[2], kernel_size=kernel_size[2], stride=stride[2])

        self.l1 = nn.Linear(in_features=self.input_shape[3], out_features=self.output_shape[3])
        self.l2 = nn.Linear(in_features=self.input_shape[4], out_features=self.output_shape[4]) 
        self.l3 = nn.Linear(in_features=self.input_shape[5], out_features=n_classes) 

    def forward(self, x):

        x = F.max_pool2d(self.conv1(x), kernel_size=self.pool_kernal[0])
        #x =  self.conv1(x)
        #x =  self.conv2(x)
        #x =  self.conv3(x)
        # x = self.conv1(x)
    
        # x = F.max_pool2d(x, kernel_size=self.pool_kernal[0])
        x = F.max_pool2d(self.conv2(x), kernel_size=self.pool_kernal[0])
        #x = F.max_pool2d(self.conv3(x), kernel_size=self.pool_kernal[1])
        x = torch.flatten(x, 1)
        # x = F.dropout(x, 0.5)
        x = F.relu(self.l1(x))
        # x = F.dropout(x, 0.3)
        x = F.relu(self.l2(x))
        # x = F.dropout(x, 0.2)
        u_alignment = x
        x = F.relu(self.l3(x))


        return x, u_alignment

class CNN_DEPTH(nn.Module):
    def __init__(self, n_classes, in_shape=[1, 32, 32, 1568, 128], out_shape=[32, 32, 32, 128], kernel_size=[7, 5, 3], stride=[2, 2, 1], pool_kernal=[2, 2, 2], pool_stride=[2, 2, 2]):
        super(CNN_DEPTH, self).__init__()
        self.pool_kernal = pool_kernal
        self.pool_stride = pool_stride

        self.input_shape = in_shape
        self.output_shape = out_shape

        self.conv1 = nn.Conv3d(in_channels=self.input_shape[0], out_channels=self.output_shape[0], kernel_size=kernel_size[0], stride=stride[0], padding=1) 
        self.bn = nn.BatchNorm3d(32)
        self.conv2 = nn.Conv3d(in_channels=self.input_shape[1], out_channels=self.output_shape[1], kernel_size=kernel_size[1], stride=stride[1], padding=1)
        self.conv3 = nn.Conv3d(in_channels=self.input_shape[2], out_channels=self.output_shape[2], kernel_size=kernel_size[2], stride=stride[2], padding=1) 

        self.l1 = nn.Linear(in_features=self.input_shape[3], out_features=self.output_shape[3])
        self.l2 = nn.Linear(in_features=self.input_shape[4], out_features=n_classes)

    def forward(self, x):
        x = torch.unsqueeze(x, 1)
        #print(x.size())
        x = F.relu(self.bn(self.conv1(x)))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal[0], stride=self.pool_stride[0])
        x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal[1], stride=self.pool_stride[1])
        x = F.relu(self.conv3(x))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal[2], stride=self.pool_stride[2])
        #print(x.size())
        x = torch.flatten(x, 1)
        x = F.dropout(x)
        x = F.relu(self.l1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.l2(x))
        return x, x

class DEPTH_LSTM(nn.Module):
    def __init__(self, n_classes, in_shape=[1, 64, 96, 96, 96, 2304, 128], out_shape=[64, 96, 96, 96, 64, 512, 128], kernel_size=[11, 5, 3, 3, 3], stride=[4, 1, 1, 1, 1], pool_kernal=[2, 2, 2, 2], pool_stride=[2, 2, 2, 2]):
        super(DEPTH_LSTM, self).__init__()
        self.pool_kernal = pool_kernal
        self.pool_stride = pool_stride

        self.input_shape = in_shape
        self.output_shape = out_shape

        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=self.output_shape[0], kernel_size=kernel_size[0], stride=stride[0], padding=1) 
        self.bn = nn.BatchNorm2d(self.output_shape[0])
        self.conv2 = nn.Conv2d(in_channels=self.input_shape[1], out_channels=self.output_shape[1], kernel_size=kernel_size[1], stride=stride[1], padding=1)
        self.conv3 = nn.Conv2d(in_channels=self.input_shape[2], out_channels=self.output_shape[2], kernel_size=kernel_size[2], stride=stride[2], padding=1) 
        self.conv4 = nn.Conv2d(in_channels=self.input_shape[3], out_channels=self.output_shape[3], kernel_size=kernel_size[3], stride=stride[3], padding=1) 
        self.conv5 = nn.Conv2d(in_channels=self.input_shape[4], out_channels=self.output_shape[4], kernel_size=kernel_size[4], stride=stride[4], padding=1) 
        #self.bn2 = nn.BatchNorm2d(self.output_shape[2])
        self.rnn = nn.LSTM(input_size=self.input_shape[5], hidden_size=self.output_shape[5], batch_first=True, bidirectional=True, num_layers=1)
        self.l1 = nn.Linear(in_features=self.output_shape[5] * 2, out_features=n_classes)

    def forward(self, x):
        batch_size = x.size()[0]
        
        x = x.flatten(0,1)
        x = torch.unsqueeze(x, 1)
        # print(x.size())
        x = F.relu(self.bn(self.conv1(x)))
        x = F.max_pool2d(x, kernel_size=self.pool_kernal[0], stride=self.pool_stride[0])
        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, kernel_size=self.pool_kernal[1], stride=self.pool_stride[1])
        x = F.relu(self.conv3(x))
        x = F.relu(self.conv4(x))
        x = F.relu(self.conv5(x))
        x = F.max_pool2d(x, kernel_size=self.pool_kernal[2], stride=self.pool_stride[2])
        # x = F.max_pool2d(x, kernel_size=self.pool_kernal[3], stride=self.pool_stride[3])
        # print(x.size())
        x = torch.flatten(x, 1)
        x = x.reshape((batch_size,60,self.input_shape[-2]))
        
        #return
        x = F.dropout(x, 0.5)

        x, (h_n, c_n) = self.rnn(x)
        x = h_n[-2:, ...]
        x = torch.transpose(x, 0, 1)
        x = torch.flatten(x, 1)

        x = F.dropout(x, 0.3)
        x = F.softmax(self.l1(x), 1)

        return x, x

class MARS(nn.Module):
    def __init__(self, n_classes, in_shape=[5, 16, 2048, 1024, 128], out_shape=[16, 32, 512, 128], kernel_size=[3, 3, 3], stride=[1, 1, 1]) -> None:
        super(MARS, self).__init__()

        self.input_shape = in_shape
        self.output_shape = out_shape        
        self.conv1 = nn.Conv2d(in_channels=self.input_shape[0], out_channels=self.output_shape[0], kernel_size=kernel_size[0], stride=stride[0], padding="same")
        self.conv2 = nn.Conv2d(in_channels=self.input_shape[1], out_channels=self.output_shape[1], kernel_size=kernel_size[1], stride=stride[1], padding="same")

        self.bn1 = nn.BatchNorm2d(self.output_shape[1], momentum=0.95)
        self.rnn = nn.LSTM(input_size=self.input_shape[2], hidden_size=self.output_shape[2], batch_first=True, bidirectional=True, num_layers=1)
        # self.l1 = nn.Linear(in_features=self.input_shape[3], out_features=self.output_shape[3])
        # self.bn2 = nn.BatchNorm1d(self.input_shape[4], momentum=0.95)
        self.l2 = nn.Linear(in_features=self.input_shape[3], out_features=n_classes)
        # self.l2 = n

    def forward(self, x):
        batch_size = x.size()[0]
        x = x.flatten(0,1)

        x = F.relu(self.conv1(x))
        x = F.dropout(x, 0.3)
        x = F.relu(self.conv2(x))
        x = F.dropout(x, 0.3)
        
        x = self.bn1(x)
        x = torch.flatten(x, 1)
        x = x.reshape((batch_size,60,self.input_shape[2]))
        
        
        x = F.dropout(x, 0.5)
        x, (h_n, c_n) = self.rnn(x)
        x = h_n[-2:, ...]
        x = torch.transpose(x, 0, 1)
        x = torch.flatten(x, 1)

        # x = F.relu(self.l1(x))
        # x = self.bn2(x)
        x = F.dropout(x, 0.4)

        x = self.l2(x)

        return x, x


class CNN_LSTM(nn.Module):
    def __init__(self, n_classes, in_shape=[1, 32, 32, 32, 32, 32, 576], out_shape=[32, 32, 32, 32, 32, 32, 64], kernel_size=[3, 3, 3], stride=[1, 1, 1], pool_kernal=[2, 2, 2], pool_stride=[2, 2, 2]):
        super(CNN_LSTM, self).__init__()

        self.pool_kernal = pool_kernal
        self.pool_stride = pool_stride

        self.input_shape = in_shape
        self.output_shape = out_shape

        self.conv1 = nn.Conv3d(in_channels=self.input_shape[0], out_channels=self.output_shape[0], kernel_size=kernel_size, stride=stride, padding="same")

        self.conv2 = nn.Conv3d(in_channels=self.input_shape[1], out_channels=self.output_shape[1], kernel_size=kernel_size, stride=stride, padding="same")

        self.conv3_1 = nn.Conv3d(in_channels=self.input_shape[2], out_channels=self.output_shape[2], kernel_size=kernel_size, stride=stride, padding="same")
        self.conv3_2 = nn.Conv3d(in_channels=self.input_shape[3], out_channels=self.output_shape[3], kernel_size=kernel_size, stride=stride, padding="same")

        self.conv3_3 = nn.Conv3d(in_channels=self.input_shape[4], out_channels=self.output_shape[4], kernel_size=kernel_size, stride=stride, padding="same")
        self.conv3_4 = nn.Conv3d(in_channels=self.input_shape[5], out_channels=self.output_shape[5], kernel_size=kernel_size, stride=stride, padding="same")

        self.rnn = nn.LSTM(input_size=self.input_shape[6], hidden_size=self.output_shape[6], batch_first=True, bidirectional=True, num_layers=1)

        self.l1 = nn.Linear(in_features=self.output_shape[6] * 2, out_features=n_classes)

    def forward(self, x)->torch.Tensor:

        batch_size = x.size()[0]
        x = x.flatten(0,1)
        x = F.relu(self.conv1(x))
        #x = F.relu(self.conv2(x))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal, stride=self.pool_stride)
        x = F.relu(self.conv3_1(x))
        x = F.relu(self.conv3_2(x))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal, stride=self.pool_stride)
        x = F.relu(self.conv3_3(x))
        #x = F.relu(self.conv3_4(x))
        x = F.max_pool3d(x, kernel_size=self.pool_kernal, stride=self.pool_stride)
        x = torch.flatten(x, 1)

        #x = x.reshape((batch_size,60,128))
        #x = x.reshape((batch_size,60,384))
        #x = x.reshape((batch_size,60,192))
        #x = x.reshape((batch_size,60,512))
        x = x.reshape((batch_size,40,self.input_shape[-1]))

        x = F.dropout(x)

        x, (h_n, c_n) = self.rnn(x)
        x = h_n[-2:, ...]

        x = torch.transpose(x, 0, 1)
        x = torch.flatten(x, 1)

        x = F.dropout(x, 0.3)
        d_alignment = x
        x = F.softmax(self.l1(x), 1)
        
        return x, d_alignment

class Fused(nn.Module):
    def __init__(self, n_classes) -> None:
        super(Fused, self).__init__()
        self.lstm_ = CNN_LSTM(n_classes)
        self.ln_ = CNN_LN(n_classes)
    
    def forward(self, x):
        # x -> [data_1, data_2]
        x_1, u = self.ln_(x[0])
        _, d = self.lstm_(x[1])

        return x_1, (u, d)


class Model(nn.Module):
    def __init__(self, n_classes, selection):
        super(Model, self).__init__()
        if selection not in ["CNN_LSTM", "MM_EAT", "CNN_LN", "Fused", "CNN_DEPTH", "DEPTH_LSTM", "MARS"]:
            raise NotImplementedError
        
        net_class = locate(f"models.{selection}")
        self.net = net_class(n_classes)
    
    def forward(self, x):
        x = self.net(x)
    
        return x

# x = torch.randn((5, 60, 10, 32, 32))
# in_shape = [60, 32, 32, 32, 32, 32, 16]
# out_shape = [32, 32, 32, 32, 32, 32, 20]
if __name__ == "__main__":
    #x = torch.randn((5, 60, 32, 128))
    x = torch.randn((5, 60, 240, 240)).to("cuda:0")

    #in_shape = [256, 96, 256, 17408, 2048, 128] # 奇怪的数字
    #out_shape = [96, 256, 512, 2048, 128]
    nt = Model(4, "DEPTH_LSTM").to("cuda:0")
    # for pm in nt.parameters():
    #     print(pm)
    nt(x)
