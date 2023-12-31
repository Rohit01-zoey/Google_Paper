import numpy as np
import random
import torch
import os
import argparse
from models.model import MobileNetV1Depth2, MobileNetV1Depth1
from models.resnet import resnet110_cifar, resnet20_cifar, resnet56_cifar, ResNet18, ResNet50, resnet110_cifarv2, resnet56_cifarv2
from models.resnet_baseline import ResNetV2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10,CIFAR100, SVHN, CelebA, ImageNet, ImageFolder
from torch.utils.data import random_split, DataLoader
from torchvision.models import resnet50, ResNet50_Weights
from torchvision.transforms import InterpolationMode
import torch.nn as nn
from torch.utils.data import Dataset


from torch import Tensor

class NormalizeTransform(torch.nn.Module):
    def __init__(self):
        super().__init__()
        
    def forward(self, x):
        x *= 2 #get it in 0 to 2 range
        x -= 1 #get it in -1 to 1 range
        return x
    
class TrDataset(Dataset):
  def __init__(self, base_dataset, transformations):
    super(TrDataset, self).__init__()
    self.base = base_dataset
    self.transformations = transformations

  def __len__(self):
    return len(self.base)

  def __getitem__(self, idx):
    x, y = self.base[idx]
    return self.transformations(x), y

# Create a custom data loader with online data augmentation
def prepare_online(ds, batch_size, shuffle=True):

    # Combine custom augmentation with other transformations as needed
    data_transforms = transforms.Compose([
        transforms.RandomHorizontalFlip(),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(0, 0, 0), interpolation=InterpolationMode.BILINEAR)
    ])

    # Apply transformations to the dataset on-the-fly
    ds = [(data_transforms(x), y) for x, y in ds]

    # Create PyTorch DataLoader
    loader = DataLoader(ds, batch_size=batch_size, shuffle=shuffle, num_workers=0, pin_memory=True, drop_last=True)

    return loader
    


def data(subset_size,data_set):
    
    print("loading the data")
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)

    if data_set == "cifar10":
        train_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeTransform(),
    
])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeTransform()
        ])
        
        train_dataset = CIFAR10(root='../data/', train=True, transform=train_transform ,download=True)
        test_dataset = CIFAR10(root='../data/', train=False, transform=test_transform, download=True)

        num_cls =10

    elif data_set == "cifar100":
        

        train_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeTransform(),
            transforms.RandomHorizontalFlip(),
            transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(0, 0, 0), interpolation=InterpolationMode.BILINEAR)
    
])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            NormalizeTransform()
        ])

        # train_dataset = CIFAR100(root='../data/', train=True, transform=train_transform ,download=True)
        train_dataset = CIFAR100(root='../data/', train=True, transform=test_transform ,download=True)
        test_dataset = CIFAR100(root='../data/', train=False, transform=test_transform, download=True)

        num_cls =100
    
    elif data_set == "svhn":
        train_transform = transforms.Compose([
    transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(0, 0, 0)),
    transforms.ToTensor()
])

        test_transform = transforms.Compose([
            transforms.ToTensor()
        ])

        train_dataset = SVHN(root='../data/', split = 'train', transform=train_transform ,download=True)
        test_dataset = SVHN(root='../data/', split = 'test', transform=test_transform, download=True)
        
        num_cls =10
    
    elif data_set == "celeb_a":
        crop_size=32
        train_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.4377, 0.4438, 0.4728), (0.1980, 0.2010, 0.1970)),
        ])

        train_dataset = CelebA(root='../data/', split = 'train', target_type = 'indentity', transform=train_transform ,download=True)
        test_dataset = CelebA(root='../data/', split = 'test', target_type = 'indentity', transform=test_transform, download=True)
        
        num_cls =10
    
    elif data_set == "tiny-imagenet":
        crop_size=64
        weights = ResNet50_Weights.IMAGENET1K_V2
        train_transform = weights.transforms() # transforms.Compose([transforms.RandomHorizontalFlip(), transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = weights.transforms()  # transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # train_dataset = ImageNet(root='../data/', split = 'train', transform=train_transform )
        # test_dataset = ImageNet(root='../data/', split = 'val', transform=test_transform)
        # Load the Tiny ImageNet training set
        train_dataset = ImageFolder('../data/tiny-imagenet-200/train', transform=train_transform)

        # Load the Tiny ImageNet validation set
        # val_dataset = ImageFolder('../data/tiny-imagenet-200/val', transform=train_transform)

        # Load the Tiny ImageNet test set
        test_dataset = ImageFolder('../data/tiny-imagenet-200/val_grouped', transform=test_transform)

        
        num_cls =200

    else:
        raise NotImplementedError
    
    if data_set in ["cifar10","cifar100"]:
        subset_length = int(len(train_dataset) * subset_size)  # if subset_size < 1, it is a ratio, otherwise it is the number of samples
    elif data_set in ["svhn","celeb_a"]:
        subset_dict = {0.15 : 7500, 0.20 : 10000, 0.25:12500, 0.30 : 15000, 0.35 : 17500 }
        subset_length = int(subset_dict[subset_size])
    elif data_set in ["tiny-imagenet"]:
        subset_dict = {0.05 : 5000, 0.10 : 10000, 0.90:90000,  1 : 100000}
        subset_length = int(subset_dict[subset_size])
    else:
        raise NotImplementedError
    
    if data_set in ["cifar10","cifar100","svhn","celeb_a"]:
        train_dataset_labelled, train_dataset_unlabelled = random_split(train_dataset, [subset_length, len(train_dataset) - subset_length])
        # train_transform = transforms.Compose([])
        # train_dataset_labelled = TrDataset(train_dataset_labelled, train_transform)
    elif data_set in ["tiny-imagenet"]:
        train_dataset_labelled, train_dataset_unlabelled = random_split(train_dataset, [subset_length, len(train_dataset) - subset_length])
    else:
        raise NotImplementedError
    
    if data_set in ["cifar10","cifar100","svhn","celeb_a"]:
        train_dataloader = DataLoader(train_dataset_labelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True, drop_last=True)
        test_dataloader = DataLoader(test_dataset, batch_size=100, shuffle=True, num_workers=0, pin_memory=True)

    elif data_set in ["tiny-imagenet"]:
        train_dataloader = DataLoader(train_dataset_labelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
        test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    print("Data loaded")
    return train_dataset_labelled, test_dataloader, num_cls

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
    train_dataloader_labelled_dataset, test_dataloader,num_cls = data(float(args.subset_size),args.data_set)
    
    if args.model == "MobileNetV1Depth2":
        model = MobileNetV1Depth2(num_classes=num_cls).to(CUDA)
    elif args.model == "MobileNetV1Depth1":
        model = MobileNetV1Depth1(num_classes=num_cls).to(CUDA)
    elif args.model == "ResNet110":
        # model = resnet110_cifarv2(num_classes=num_cls).to(CUDA)
        # Create the ResNetV2 model
        depth = 29
        in_planes = 3
        num_classes = 100
        data_augmentation = False
        model = ResNetV2(depth, in_planes, num_classes, data_augmentation).to(CUDA)
    elif args.model == "ResNet20":
        model = resnet20_cifar(num_classes=num_cls).to(CUDA)
    elif args.model == "ResNet56":
        model = resnet56_cifarv2(num_classes=num_cls).to(CUDA)
    elif args.model == "ResNet18":
        model = ResNet18(num_classes=num_cls).to(CUDA)
    elif args.model == "ResNet50":
        model = resnet50(weights=ResNet50_Weights.IMAGENET1K_V2).to(CUDA)
        model.fc = nn.Linear(2048, num_cls).to(CUDA)
        #model = ResNet50(num_classes=num_cls).to(CUDA)
    
        
    criterion = torch.nn.CrossEntropyLoss().to(CUDA)
    
    if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
        # optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-2)
        optimizer = torch.optim.Adam(model.parameters(), lr=1e-3, weight_decay=1e-4) 
    elif args.data_set in ['tiny-imagenet']:
        #optimizer =torch.optim.SGD(model.parameters() , lr=0.1,momentum=0.9,weight_decay=5e-4, nesterov=True)
        optimizer =torch.optim.SGD(model.parameters() , lr=0.1,momentum=0.2, nesterov=False)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epoch//4)

    DIR = "./results/"+args.data_set+"/"+args.subset_size+"/"+args.model
    
    os.makedirs(DIR, exist_ok = True)

    BEST_PATH = DIR+"/best_model.pt"
    LAST_PATH = DIR+"/last_model.pt"
    log_file = open(DIR+"/output.txt", 'w')

    if args.checkpoint:
        checkpoint = torch.load(LAST_PATH)
        model.load_state_dict(checkpoint['state_dict'],map_location='cuda:0')
        epoch_ = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        
    else:
        epoch_ = 0
    
    print(f"Training with {len(train_dataloader_labelled_dataset.dataset)} point")
    max_test_acc = 0
    for epoch in range(epoch_,args.epoch):
        print(f"Starting epoch: {epoch}",file=log_file)
        train_loss = 0.0
        train_acc = 0.0
        test_loss = 0.0
        test_acc = 0.0
        iteration = 0
        best_loss = float('inf')
        
        train_dataloader_labelled = prepare_online(train_dataloader_labelled_dataset, batch_size=128, shuffle=True)

        if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
            for group in optimizer.param_groups:
                group['lr'] = 1e-3
            if(epoch>=180):
                for group in optimizer.param_groups:
                    group['lr'] *= 0.5e-3 # 0.5
            elif(epoch>=160):
                for group in optimizer.param_groups:
                    group['lr'] *= 1e-3 # 1e-1
            elif(epoch>=120):
                for group in optimizer.param_groups:
                    group['lr'] *= 1e-2 # 1e-1
            elif(epoch>=80): 
                for group in optimizer.param_groups:
                    group['lr'] *= 1e-1
                    
        elif args.data_set in ['tiny-imagenet']:
            if(epoch>=80):
                for group in optimizer.param_groups:
                    group['lr'] = 0.0001
            elif(epoch>=60):
                for group in optimizer.param_groups:
                    group['lr'] = 0.001
            elif(epoch>=30):
                for group in optimizer.param_groups:
                    group['lr'] = 0.01
        else:
            raise NotImplementedError
         

        # Training...  
        model.train()     
        for input, label in train_dataloader_labelled:
            input, label = input.to(device=CUDA), label.to(device=CUDA)
            output = model(input)
            output_idx = torch.argmax(output, dim=1)
            train_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion(output, label)
            train_loss += loss.item()
            iteration += 1
            
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            #scheduler.step(epoch + iteration/len(train_dataloader_labelled))
            
        print("| Train Loss | Train Acc   | {a:.4f} | {b:.4f} | lr = {c}".format(a=train_loss/iteration, b=train_acc.item()/iteration, c = [group['lr'] for group in optimizer.param_groups]),flush=True,file=log_file)  
        
        # Testing
        iteration=0
        model.eval()
        for input, label in test_dataloader:
            input, label = input.to(device=CUDA), label.to(device=CUDA)
            with torch.inference_mode():
                output = model(input)
            output = output.detach()
            output_idx = torch.argmax(output, dim=1)
            test_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion(output, label)
            test_loss += loss.item()
            iteration += 1
        max_test_acc = max(max_test_acc, test_acc.item()/iteration)
        print("| Test Loss  | Test Acc    | {a:.4f} | {b:.4f} |".format(a=test_loss/iteration, b=test_acc.item()/iteration), flush=True,file=log_file)
        print("\n",file=log_file)
        if(best_loss > test_loss):
            best_loss = test_loss
            torch.save(model.state_dict(), BEST_PATH)
        
        ckpt_state = {'epoch': epoch + 1, 'state_dict': model.state_dict(),\
                      'optimizer': optimizer.state_dict()}#,'scheduler': scheduler.state_dict()}
        torch.save(ckpt_state, LAST_PATH)
        
    print("Best Test Acc: ", max_test_acc,file=log_file)     
    log_file.close()
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    
    parser.add_argument('--subset_size', '-s', type=str, help="how much labelled dataset to use, for e.g., 10 percent is 0.1")
    # parser.add_argument('--mcd_loss', )
    # parser.add_argument('--teacher_or_student', '-t', help="to train teacher use 1 for student use 0")
    # parser.add_argument('--alpha', '-a', help="unlabelled loss weightage")
    parser.add_argument('--model', '-m', type=str)
    parser.add_argument('--data_set', '-d', type=str,default="cifar100")
    parser.add_argument('--seed', '-se', type=int,default=753410)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--cuda', '-c', type=str, help='which gpu to use',default='0')
    parser.add_argument('--checkpoint', '-ch', type=bool, default=False)
    #parser.add_argument('--teacher', '-t', type=int, default=1)
    args = parser.parse_args()
    
    main(args)
