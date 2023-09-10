"""
resnet for cifar in pytorch
Reference:
[1] K. He, X. Zhang, S. Ren, and J. Sun. Deep residual learning for image recognition. In CVPR, 2016.
[2] K. He, X. Zhang, S. Ren, and J. Sun. Identity mappings in deep residual networks. In ECCV, 2016.
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import math


def conv3x3(in_planes, out_planes, stride=1):
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)


class BasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(BasicBlock, self).__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        # residual = self.conv1(x)
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample is not None:
            residual = self.downsample(x)
        out += residual
        out = self.relu(out)
        return out


class Bottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn1 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes * 4)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            residual = self.downsample(x)

        out += residual
        out = self.relu(out)

        return out


class PreActBasicBlock(nn.Module):
    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBasicBlock, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = conv3x3(planes, planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out += residual

        return out


class PreActBottleneck(nn.Module):
    expansion = 4

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super(PreActBottleneck, self).__init__()
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv1 = nn.Conv2d(inplanes, planes, kernel_size=1, bias=False)
        self.bn2 = nn.BatchNorm2d(planes)
        self.conv2 = nn.Conv2d(planes, planes, kernel_size=3, stride=stride, padding=1, bias=False)
        self.bn3 = nn.BatchNorm2d(planes)
        self.conv3 = nn.Conv2d(planes, planes * 4, kernel_size=1, bias=False)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        residual = x

        out = self.bn1(x)
        out = self.relu(out)

        if self.downsample is not None:
            residual = self.downsample(out)

        out = self.conv1(out)

        out = self.bn2(out)
        out = self.relu(out)
        out = self.conv2(out)

        out = self.bn3(out)
        out = self.relu(out)
        out = self.conv3(out)

        out += residual

        return out


class ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10,if_large=False):
        super(ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.relu = nn.ReLU(inplace=True)
        self.layer1 = self._make_layer(block, 16, layers[0]) # no strides set
        self.layer2 = self._make_layer(block, 32, layers[1], stride=2) # stride = 2
        self.layer3 = self._make_layer(block, 64, layers[2], stride=2) # stride = 2
        if if_large:
            self.avgpool = nn.AvgPool2d(8*3, stride=1)
        else:
            self.avgpool = nn.AvgPool2d(8, stride=1)
        
        self.fc = nn.Linear(64 * block.expansion, num_classes)

        self.embDim = 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):

        if freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)
                x = self.relu(x)

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                x = self.avgpool(x)
                e = x.view(x.size(0), -1)

        else:
            x = self.conv1(x)
            x = self.bn1(x)
            x = self.relu(x)


            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)
            x = self.avgpool(x)
            e = x.view(x.size(0), -1)

        x = self.fc(e)
        
        if last:
            return x, e
        else:
            return x

    def get_embedding_dim(self):
        return self.embDim


class PreAct_ResNet_Cifar(nn.Module):

    def __init__(self, block, layers, num_classes=10):
        super(PreAct_ResNet_Cifar, self).__init__()
        self.inplanes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16) #! not there in og code
        self.layer1 = self._make_layer(block, 16, layers[0]) # channels = 16
        self.layer2 = self._make_layer(block, 128, layers[1], stride=2) # 2 normally, channels = 32
        self.layer3 = self._make_layer(block, 256, layers[2], stride=2) # channels = 32
        self.bn = nn.BatchNorm2d(256 * block.expansion) # og was 64 * block.expansion
        self.relu = nn.ReLU(inplace=True)
        self.avgpool = nn.AvgPool2d(8, stride=1)
        self.fc = nn.Linear(256 * block.expansion, num_classes) # og was 64 * block.expansion

        self.embDim = 256 * block.expansion # og was 64 * block.expansion

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                n = m.kernel_size[0] * m.kernel_size[1] * m.out_channels
                m.weight.data.normal_(0, math.sqrt(2. / n))
            elif isinstance(m, nn.BatchNorm2d):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.inplanes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion) #! not there in og code
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))
        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):

        if freeze:
            with torch.no_grad():
                x = self.conv1(x)
                x = self.bn1(x)#! not there in og code
                x = self.relu(x) #! not there in og code

                x = self.layer1(x)
                x = self.layer2(x)
                x = self.layer3(x)

                x = self.bn(x)
                x = self.relu(x)
                x = self.avgpool(x)
                e = x.view(x.size(0), -1)

        else:

            x = self.conv1(x)
            x = self.bn1(x)#! not there in og code
            x = self.relu(x)#! not there in og code
            
            x = self.layer1(x)
            x = self.layer2(x)
            x = self.layer3(x)

            x = self.bn(x)
            x = self.relu(x)
            x = self.avgpool(x)
            e = x.view(x.size(0), -1)

        x = self.fc(e)

        if last:
            return x, e
        else:
            return x

    def get_embedding_dim(self):
        return self.embDim


def resnet14_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet8_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [1, 1, 1], **kwargs)
    return model


def resnet20_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model

def resnet20_cifarv2(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet26_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [4, 4, 4], **kwargs)
    return model


def resnet32_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet44_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet56_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model

def resnet56_cifarv2(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet110_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model

def resnet110_cifarv2(**kwargs):
    model = PreAct_ResNet_Cifar(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet1202_cifar(**kwargs):
    model = ResNet_Cifar(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet164_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet1001_cifar(**kwargs):
    model = ResNet_Cifar(Bottleneck, [111, 111, 111], **kwargs)
    return model


def preact_resnet110_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBasicBlock, [18, 18, 18], **kwargs)
    return model


def preact_resnet164_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [18, 18, 18], **kwargs)
    return model


def preact_resnet1001_cifar(**kwargs):
    model = PreAct_ResNet_Cifar(PreActBottleneck, [111, 111, 111], **kwargs)
    return model



        
class ResNetMetaNet(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[5, 5, 5], use_sigmoid=False):
        super(ResNetMetaNet, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.linear = nn.Linear(64, 2)
        # self.apply(_weights_init) # here we intiliaze the weights later in the main code not here
    def _make_layer(self, block, planes, num_blocks, stride):
        strides = [stride] + [1]*(num_blocks-1)
        layers = []
        for stride in strides:
            layers.append(block(self.in_planes, planes, stride))
            self.in_planes = planes * block.expansion
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
    
class ResNetMetaNetv2(nn.Module):
    def __init__(self, block=BasicBlock, num_blocks=[5, 5, 5], use_sigmoid=False):
        super(ResNetMetaNetv2, self).__init__()
        self.in_planes = 16
        self.conv1 = nn.Conv2d(3, 16, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(16)
        self.layer1 = self._make_layer(block, 16, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 32, num_blocks[1], stride=1)
        self.layer3 = self._make_layer(block, 64, num_blocks[2], stride=1)
        self.linear = nn.Linear(64, 2)
        # self.apply(_weights_init) # here we intiliaze the weights later in the main code not here
    def _make_layer(self, block, planes, blocks, stride):
        # strides = [stride] + [1]*(num_blocks-1)
        # layers = []
        # for stride in strides:
        #     layers.append(block(self.in_planes, planes, stride))
        #     self.in_planes = planes * block.expansion
        # return nn.Sequential(*layers)
    
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion) #! not there in og code
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))
        return nn.Sequential(*layers)
    def forward(self, x):
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.layer1(out)
        out = self.layer2(out)
        out = self.layer3(out)
        out = F.avg_pool2d(out, out.size()[3])
        out = out.view(out.size(0), -1)
        out = self.linear(out)
        out = torch.sigmoid(out)
        return out
        
def resnet_metanet_14_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [2, 2, 2], **kwargs)
    return model


def resnet_metanet_8_cifar(**kwargs):
    model = ResNetMetaNetv2(BasicBlock, [1, 1, 1], **kwargs)
    return model


def resnet_metanet_20_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [3, 3, 3], **kwargs)
    return model


def resnet_metanet_26_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [4, 4, 4], **kwargs)
    return model


def resnet_metanet_32_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [5, 5, 5], **kwargs)
    return model


def resnet_metanet_44_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [7, 7, 7], **kwargs)
    return model


def resnet_metanet_56_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [9, 9, 9], **kwargs)
    return model


def resnet_metanet_110_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [18, 18, 18], **kwargs)
    return model


def resnet_metanet_1202_cifar(**kwargs):
    model = ResNetMetaNet(BasicBlock, [200, 200, 200], **kwargs)
    return model


def resnet_metanet_164_cifar(**kwargs):
    model = ResNetMetaNet(Bottleneck, [18, 18, 18], **kwargs)
    return model


def resnet_metanet_1001_cifar(**kwargs):
    model = ResNetMetaNet(Bottleneck, [111, 111, 111], **kwargs)
    return model

        

resnet_book = {
    '8': resnet8_cifar,
    '14': resnet14_cifar,
    '20': resnet20_cifar,
    '26': resnet26_cifar,
    '32': resnet32_cifar,
    '44': resnet44_cifar,
    '56': resnet56_cifar,
    '110': resnet110_cifar,
}


resnet_metanet_book = {
    '8': resnet_metanet_8_cifar,
    '14': resnet_metanet_14_cifar,
    '20': resnet_metanet_20_cifar,
    '26': resnet_metanet_26_cifar,
    '32': resnet_metanet_32_cifar,
    '44': resnet_metanet_44_cifar,
    '56': resnet_metanet_56_cifar,
    '110': resnet_metanet_110_cifar,
}




####Extra code added from distil code base
class ResNet(nn.Module):
    def __init__(self, block, num_blocks, num_classes=10, channels=3):
        super(ResNet, self).__init__()
        self.in_planes = 64
        self.embDim = 8 * self.in_planes * block.expansion
        
        self.conv1 = nn.Conv2d(channels, 64, kernel_size=3, stride=1, padding=1, bias=False)
        self.bn1 = nn.BatchNorm2d(64)
        self.layer1 = self._make_layer(block, 64, num_blocks[0], stride=1)
        self.layer2 = self._make_layer(block, 128, num_blocks[1], stride=2)
        self.layer3 = self._make_layer(block, 256, num_blocks[2], stride=2) # stride = 2
        self.layer4 = self._make_layer(block, 512, num_blocks[3], stride=3) # stride = 2
        self.linear = nn.Linear(512*block.expansion, num_classes)


    # def _make_layer(self, block, planes, num_blocks, stride):
    #     strides = [stride] + [1]*(num_blocks-1)
    #     layers = []
    #     for stride in strides:
    #         layers.append(block(self.in_planes, planes, stride))
    #         self.in_planes = planes * block.expansion
    #     return nn.Sequential(*layers)
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.in_planes != planes * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_planes, planes * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(planes * block.expansion)
            )

        layers = []
        layers.append(block(self.in_planes, planes, stride, downsample))
        self.in_planes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.in_planes, planes))

        return nn.Sequential(*layers)

    def forward(self, x, last=False, freeze=False):
        if freeze:
            with torch.no_grad():
                out = F.relu(self.bn1(self.conv1(x)))
                out = self.layer1(out)
                out = self.layer2(out)
                out = self.layer3(out)
                out = self.layer4(out)
                out = F.avg_pool2d(out, 4)
                e = out.view(out.size(0), -1)
        else:
            out = F.relu(self.bn1(self.conv1(x)))
            # print("Line468: {}".format(out.shape))
            out = self.layer1(out)
            # print("Line470: {}".format(out.shape))
            out = self.layer2(out)
            # print("Line472: {}".format(out.shape))
            out = self.layer3(out)
            # print("Line474: {}".format(out.shape))
            out = self.layer4(out)
            # print("Line476: {}".format(out.shape))
            out = F.avg_pool2d(out, 4)
            # print("Line478: {}".format(out.shape))
            e = out.view(out.size(0), -1)
            # print("Line480: {}".format(e.shape))
        out = self.linear(e)
        if last:
            return out, e
        else:
            return out

    def get_embedding_dim(self):
        return self.embDim
    
    
def ResNet18(num_classes=10, channels=3):
    return ResNet(BasicBlock, [2,2,2,2], num_classes, channels)


def ResNet34(num_classes=10, channels=3):
    return ResNet(BasicBlock, [3,4,6,3], num_classes, channels)


def ResNet50(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,4,6,3], num_classes, channels)


def ResNet101(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,4,23,3], num_classes, channels)


def ResNet152(num_classes=10, channels=3):
    return ResNet(Bottleneck, [3,8,36,3], num_classes, channels)