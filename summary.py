from models.resnet import resnet110_cifar, resnet20_cifar, resnet20_cifarv2
from models.resnet_baseline import ResNetV2, ResNetLayer
from torchsummary import summary
import numpy as np
import random
import torch
import os
import argparse



def set_seed(seed=2):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed) 

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda
    set_seed(args.seed)
    if args.model == 'resnet20':
        model = resnet20_cifarv2(num_classes=100).to(CUDA)
    elif args.model == 'resnet20v2':
        model = ResNetV2(depth=20, num_classes=100).to(CUDA)
    elif args.model == 'res_layer':
        model = ResNetLayer(in_channels=64, out_channels=16, kernel_size=1, strides=1, activation='relu', batch_normalization=True, conv_first=False).to(CUDA)
        
    summary_text = summary(model, (3, 32, 32))
    
    # Save the summary to a text file
    # models/
    # with open(f"models/model_summary_{args.model}.txt", "w") as text_file:
    #     text_file.write(summary_text)
    # text_file.close()
        
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', '-s', type=int, default=2)
    parser.add_argument('--model', '-m', type=str, default='resnet20')
    parser.add_argument('--cuda', '-c', type=str, help='which gpu to use',default='0')
    #parser.add_argument('--teacher', '-t', type=int, default=1)
    args = parser.parse_args()
    
    main(args)