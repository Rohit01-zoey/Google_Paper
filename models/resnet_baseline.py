import torch
import torch.nn as nn
import torch.nn.functional as F

class ResNetLayer(nn.Module):
    def __init__(self, in_channels, out_channels=16, kernel_size=3, strides=1, activation='relu', batch_normalization=True, conv_first=True):
        super(ResNetLayer, self).__init__()
        
        padding = (kernel_size - 1) // 2 # 'same' padding since kernel size is odd we do floor division
        layers = []
        if conv_first:
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=True))
            if batch_normalization:
                layers.append(nn.BatchNorm2d(out_channels))
            if activation is not None:
                layers.append(nn.ReLU(inplace=True))
        else:
            if batch_normalization:
                layers.append(nn.BatchNorm2d(in_channels))
            if activation is not None:
                layers.append(nn.ReLU(inplace=True))
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size=kernel_size, stride=strides, padding=padding, bias=True))

        self.layer = nn.Sequential(*layers)

    def forward(self, x):
        return self.layer(x)


class ResNetV2(nn.Module):
    def __init__(self, depth, in_planes = 3, num_classes=10, data_augmentation=False):
        super(ResNetV2, self).__init__()

        if (depth - 2) % 9 != 0:
            raise ValueError('depth should be 9n+2 (eg 56 or 110 in [b])')

        num_filters_in = 16
        prev_num_filters = num_filters_in
        self.num_res_blocks = int((depth - 2) / 9)

        self.data_augmentation = data_augmentation

        if self.data_augmentation:
            self.data_augmentation_module = nn.Sequential(
                nn.RandomHorizontalFlip(),
                nn.RandomAffine(degrees=0, translate=(3./32, 3./32))
            )

        self.conv_first = ResNetLayer(in_planes, num_filters_in, conv_first=True)

        self.res_blocks = nn.ModuleList()
        for stage in range(3):
            for res_block_iter in range(self.num_res_blocks):
                activation = 'relu'
                batch_normalization = True
                strides = 1
                if stage == 0:
                    num_filters_out = num_filters_in * 4
                    if res_block_iter == 0:  # first layer and first stage
                        activation = None
                        batch_normalization = False
                else:
                    num_filters_out = num_filters_in * 2
                    if res_block_iter == 0:  # first layer but not first stage
                        strides = 2  # downsample

                bottleneck_block = nn.Sequential(
                    ResNetLayer(prev_num_filters, num_filters_in, kernel_size=1, strides=strides, activation=activation, batch_normalization=batch_normalization, conv_first=False),
                    ResNetLayer(num_filters_in, num_filters_in, kernel_size=3, strides=1, conv_first=False),
                    ResNetLayer(num_filters_in, num_filters_out, kernel_size=1, strides=1, conv_first=False)
                )

                self.res_blocks.append(bottleneck_block)

                if res_block_iter == 0:
                    # linear projection residual shortcut connection to match changed dims
                    projection_block = ResNetLayer(num_filters_in, num_filters_out, kernel_size=1, strides=strides, activation=None, batch_normalization=False)
                    self.res_blocks.append(projection_block)
                
                prev_num_filters = num_filters_out # append the final output of the layer to the input of the next layer

            num_filters_in = num_filters_out

        self.classifier = nn.Sequential(
            nn.BatchNorm2d(num_filters_out),
            nn.ReLU(inplace=True),
            nn.AdaptiveAvgPool2d(1),
            nn.Flatten(),
            nn.Linear(num_filters_out, num_classes)
        )

    def forward(self, x):
        if self.data_augmentation:
            x = self.data_augmentation_module(x)

        x = self.conv_first(x)

        block_idx = 0
        for stage in range(3):
            for res_block in range(self.num_res_blocks):
                x_input = x
                y = self.res_blocks[block_idx](x)
                block_idx += 1
                if res_block == 0:
                    x = self.res_blocks[block_idx](x_input)
                    block_idx += 1
                x = y + x # short cut connection

        x = self.classifier(x)

        return x


