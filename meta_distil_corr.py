import numpy as np
import random
import torch
import os
import argparse
from models.model import InstanceMetaNet,MobileNetV1Depth2, MobileNetV1Depth1
from models.resnet import resnet110_cifar, resnet20_cifar, resnet_metanet_8_cifar, resnet56_cifar, ResNet18, ResNet50, resnet110_cifarv2, resnet56_cifarv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, CelebA, ImageFolder
from torch.utils.data import random_split, DataLoader
from meta import MetaSGD
import torch.nn.functional as F

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

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

def mcd_loss(net, input, n_evals=5):
    mc_samples = [net(input) for _ in range(n_evals)]
    mc_samples = torch.stack(mc_samples) #(n_evals, B, classes)
    std_pred = torch.std(mc_samples, dim=0) #(B, classes)
    std_pred = torch.sum(std_pred)/(input.shape[0]*mc_samples.shape[-1])
    return std_pred

def data(subset_size,data_set):
    
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    if data_set == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomHorizontalFlip(),
    transforms.RandomAffine(degrees=0, translate=(0.1, 0.1), fill=(0, 0, 0)),
    transforms.ToTensor()
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor()
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

        train_dataset = CIFAR100(root='../data/', train=True, transform=train_transform ,download=True)
        test_dataset = CIFAR100(root='../data/', train=False, transform=test_transform, download=True)

        num_cls =100
        
        
    elif data_set == "svhn":
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
        crop_size=32
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(),  transforms.ToTensor()])
        test_transform = transforms.Compose([transforms.RandomHorizontalFlip(),  transforms.ToTensor()]) # transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
        # train_dataset = ImageNet(root='../data/', split = 'train', transform=train_transform )
        # test_dataset = ImageNet(root='../data/', split = 'val', transform=test_transform)
        # Load the Tiny ImageNet training set
        train_dataset = ImageFolder('../data/tiny-imagenet-200/train', transform=train_transform)

        # Load the Tiny ImageNet validation set
        #  val_dataset = ImageFolder('../data/tiny-imagenet-200/val', transform=train_transform)

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
        subset_dict = {0.05 : 5000, 0.10 : 10000, 0.90 : 90000}
        subset_length = int(subset_dict[subset_size])
    train_dataset_labelled, train_dataset_unlabelled = random_split(train_dataset, [subset_length, len(train_dataset) - subset_length])
    meta_dataset, val_dataset, train_dataset_unlabelled = random_split(train_dataset_unlabelled, [1000, 2000, len(train_dataset_unlabelled) - 3000])

    train_dataloader_l = DataLoader(train_dataset_labelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True) # int((64 * (1.0-subset_size))/subset_size)+1
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    meta_dataloader = DataLoader(meta_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    val_dataloader = DataLoader(val_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_dataloader_l, train_dataloader_u, meta_dataloader, test_dataloader, val_dataloader, num_cls

def set_seed(seed=0):
    np.random.seed(seed)
    random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    # When running on the CuDNN backend, two further options must be set
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    # Set a fixed value for the hash seed
    os.environ["PYTHONHASHSEED"] = str(seed) 

def get_model(model_name,num_cls):
    if model_name == "MobileNetV1Depth2":
        model = MobileNetV1Depth2(num_classes=num_cls).to(CUDA)
    elif model_name == "MobileNetV1Depth1":
        model = MobileNetV1Depth1(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet110":
        model = resnet110_cifarv2(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet20":
        model = resnet20_cifar(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet56":
        model = resnet56_cifarv2(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes=num_cls).to(CUDA)
    
    return model

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda
    set_seed(args.seed)
        
    train_dataloader_l, train_dataloader_u,meta_dataloader, test_dataloader,val_dataloader,num_cls = data(float(args.subset_size),args.data_set)

    # meta_net = InstanceMetaNet().to(CUDA)
    meta_net = resnet_metanet_8_cifar().to(CUDA) # using the updated revar code for the meta net

    student = get_model(args.stu_model,num_cls)
    teacher = get_model(args.teacher_model,num_cls)
    teacher.load_state_dict(torch.load(\
        "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+'/best_model.pt',map_location='cuda:0'))
    teacher.eval()
    
    criterion_KD = torch.nn.KLDivLoss(reduction='none').to(CUDA)
    criterion_CE = torch.nn.CrossEntropyLoss().to(CUDA)
    
    if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay = 1e-4)
    elif args.data_set in ['tiny-imagenet']:
        optimizer =torch.optim.SGD(student.parameters() , lr=0.1,momentum=0.9,weight_decay=5e-4, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epoch//4)
    
    meta_optimizer = torch.optim.Adam(meta_net.parameters(), lr = args.meta_lr, weight_decay=args.meta_weight_decay)
    #meta_optimizer =torch.optim.SGD(meta_net.parameters() , lr=0.1,momentum=0.9,weight_decay=5e-4, nesterov=True)
    #meta_scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(meta_optimizer, args.epoch//4)
    #torch.optim.lr_scheduler.CosineAnnealingLR(meta_optimizer, args.epoch)
    
    DIR = "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+"_"+args.stu_model+"/meta_distil"
    
    os.makedirs(DIR, exist_ok = True)

    BEST_PATH = DIR+"/best_model.pt"
    LAST_PATH = DIR+"/last_model.pt"
    log_file = open(DIR+f"/output_mlr_{args.meta_lr}_mwd_{args.meta_weight_decay}.txt", 'w')

    if args.checkpoint:
        checkpoint = torch.load(LAST_PATH)
        student.load_state_dict(checkpoint['state_dict'],map_location='cuda:0')
        epoch_ = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
        
        meta_net.load_state_dict(checkpoint['meta-net-state'],map_location='cuda:0')
        meta_optimizer.load_state_dict(checkpoint['meta-optimizer'])
        #meta_scheduler.load_state_dict(checkpoint['meta-scheduler'])
    else:
        epoch_ = 0
    
    max_test_acc = 0.0
    max_val_acc = 0.0
    
    meta_dataloader_iter = iter(meta_dataloader)

    Temp = args.temp
    
    for epoch in range(epoch_,args.epoch):
        print("Starting epoch: ", epoch, flush=True,file=log_file)
        train_loss = 0.0
        train_loss_u = 0.0
        val_loss = 0.0
        test_loss = 0.0
        train_acc = 0.0
        train_acc_u = 0.0
        total_loss = 0.0
        val_acc=0.0
        test_acc = 0.0
        iteration = 0
        total_meta_loss = 0.0
        meta_acc = 0.0
        best_loss = float('inf')
        best_val_loss = float('inf')
        
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
        student.train()
        
        for (input, label), (input_u, label_u) in zip(train_dataloader_l, train_dataloader_u):
            # Labelled
            input, label = input.to(CUDA), label.to(CUDA)
            output = student(input)
            output_idx = torch.argmax(output, dim=1)
            train_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            train_loss += loss.item()
            
            input_u, label_u = input_u.to(CUDA), label_u.to(CUDA)
            
            # Run Meta Network thing
            pseudo_net = get_model(args.stu_model,num_cls)
            pseudo_net.load_state_dict(student.state_dict())
            
            pseudo_outputs = pseudo_net(input_u)
            with torch.no_grad():
                t_outputs = teacher(input_u)
            t_outputs = F.softmax(t_outputs/Temp, dim=1)
            pseudo_loss_vector = criterion_KD(F.log_softmax(pseudo_outputs/Temp, dim=1), t_outputs)
            #pseudo_loss_vector_reshape = torch.reshape(pseudo_loss_vector, (-1, 1))
            
            pseudo_weight = meta_net(input_u)#*2
            #pseudo_weight_no_norm = meta_net(input_u)
            #pseudo_weight = (pseudo_weight_no_norm - pseudo_weight_no_norm.min())/(pseudo_weight_no_norm.max() - pseudo_weight_no_norm.min()+1e-14)
            #pseudo_weight = pseudo_weight_no_norm/(pseudo_weight_no_norm.sum()+1e-14) 
            
            pseudo_loss = (Temp**2)*torch.mean(pseudo_weight[...,0] * torch.sum(pseudo_loss_vector,dim=1))
            #pseudo_loss = (Temp**2)*torch.sum(pseudo_weight.view(-1) * torch.sum(pseudo_loss_vector,dim=1))
            
            pseudo_grad = torch.autograd.grad(pseudo_loss, pseudo_net.parameters(), create_graph = True)
            
            pseudo_optimizer = MetaSGD(pseudo_net, pseudo_net.parameters(), lr = 1e-3)
            
            pseudo_optimizer.load_state_dict(optimizer.state_dict())
            pseudo_optimizer.meta_step(pseudo_grad)
            
            del pseudo_grad
            
            # Meta Updates
            try:
                meta_inputs, meta_labels = next(meta_dataloader_iter)
            except StopIteration:
                meta_dataloader_iter = iter(meta_dataloader)
                meta_inputs, meta_labels = next(meta_dataloader_iter)
                
            meta_inputs, meta_labels = meta_inputs.to(CUDA), meta_labels.to(CUDA)
            meta_outputs = pseudo_net(meta_inputs)
            
            meta_loss = criterion_CE(meta_outputs, meta_labels.long()) 
            #+ args.mcd_weight*mcd_loss(pseudo_net, input_u)
            #+ args.mcd_weight*mcd_loss(pseudo_net, meta_inputs)
            #
            
            #total_meta_loss += meta_loss.item()
             
            meta_optimizer.zero_grad()
            meta_loss.backward()
            meta_optimizer.step()
            #meta_scheduler.step(epoch + iteration/len(train_dataloader_u))
            
            # Unlabelled
            output_u = student(input_u)
            output_idx_u = torch.argmax(output_u, dim=1)
            train_acc_u += torch.sum(output_idx_u == label_u)/len(input_u)
            with torch.inference_mode():
                logits = teacher(input_u)
            logits = logits.detach().clone()
            
            soft_targets = torch.nn.functional.softmax(logits/Temp, dim=1)
            loss_u = torch.sum(criterion_KD(torch.log_softmax(output_u/Temp, dim=1), soft_targets),dim=1)
            train_loss_u += (Temp**2)*torch.mean(loss_u).item()
            
            with torch.no_grad():
                #weight_no_norm = meta_net(input_u)
                #weight = (weight_no_norm - weight_no_norm.min())*2/(weight_no_norm.max() - weight_no_norm.min()+1e-14)
                #weight = weight_no_norm/(weight_no_norm.sum()+1e-14) 
                
                weight = meta_net(input_u)

            if iteration == 0:
                weights = weight.detach().cpu()
            else:
                weights = torch.cat((weights, weight.detach().cpu()),dim =0)
            
            total_loss = loss + (Temp**2)*torch.mean(weight[...,0] * loss_u )
            #total_loss = loss + (Temp**2)*torch.sum(weight.view(-1) * loss_u )
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step() 
            #scheduler.step(epoch + iteration/len(train_dataloader_u))
            
            iteration += 1
        print("| Labelled   Loss | Labelled   Acc | {a:.4f} | {b:.4f} |".format(a=train_loss/iteration, b=train_acc.item()/iteration), flush=True,file=log_file)  
        print("| Unlabelled Loss | Unlabelled Acc | {a:.4f} | {b:.4f} |".format(a=train_loss_u/iteration, b=train_acc_u.item()/iteration), flush=True,file=log_file)
        print("| Total      Loss | Total      Acc | {a:.4f} | {b:.4f} |".format(a=total_loss/iteration, b=(train_acc * float(args.subset_size) + train_acc_u * (1.0 - float(args.subset_size)))/iteration), flush=True,file=log_file)
        
        print("Weights",torch.quantile(weights,0.25),torch.quantile(weights,0.5),torch.quantile(weights,0.75),file=log_file)
        
        student.eval()

        iteration = 0
        for input, label in meta_dataloader:
            input, label = input.to(CUDA), label.to(CUDA)
            with torch.inference_mode():
                output = student(input)
            output = output.detach()
            output_idx = torch.argmax(output, dim=1)
            meta_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            total_meta_loss += loss.item()
            iteration += 1

        print("| Meta Loss  |  Meta  Acc |{a:.4f} | {b:.4f} |".format(a=total_meta_loss/iteration, b=meta_acc.item()/iteration), flush=True,file=log_file) 
        
        
        # validation accuracy and loss computation
        iteration = 0
        for input, label in val_dataloader:
            input, label = input.to(CUDA), label.to(CUDA)
            with torch.inference_mode():
                output = student(input)
            output = output.detach()
            output_idx = torch.argmax(output, dim=1)
            val_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            val_loss += loss.item()
            iteration += 1

        print("| Val       Loss | Val       Acc | {a:.4f} | {b:.4f} |".format(a=val_loss/iteration, b=val_acc.item()/iteration), flush=True,file=log_file)
        
        max_val_acc = max(max_val_acc, val_acc.item()/iteration)
    #     print("| Test Loss  | Test Acc    | {a:.4f} | {b:.4f} |".format(a=test_loss/iteration, b=test_acc.item()/iteration))
        
        if(best_val_loss > val_loss):
            best_val_loss = val_loss
        
        # ckpt_state = {'epoch': epoch + 1, 'state_dict': student.state_dict(),
        #               'optimizer': optimizer.state_dict(),'meta-net-state':meta_net.state_dict(),
        #               'meta-optimizer': meta_optimizer.state_dict()}#,'meta-scheduler': meta_scheduler.state_dict()}
        # #,'scheduler': scheduler.state_dict()
        
        # torch.save(ckpt_state, LAST_PATH)
        
        print('Best Val Acc: ', max_val_acc,file=log_file)
        print("\n",file=log_file)
        
        # test accuracy and loss calculation
        iteration = 0
        for input, label in test_dataloader:
            input, label = input.to(CUDA), label.to(CUDA)
            with torch.inference_mode():
                output = student(input)
            output = output.detach()
            output_idx = torch.argmax(output, dim=1)
            test_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            test_loss += loss.item()
            iteration += 1

        print("| Test       Loss | Test       Acc | {a:.4f} | {b:.4f} |".format(a=test_loss/iteration, b=test_acc.item()/iteration), flush=True,file=log_file)
        
    # print()   
    # print("*"*50)
    # print("*"*50)
    # print("*"*50)
    # print()
    
    # for group in optimizer.param_groups:
    #     group['lr'] = 1e-5
    
    # fine tuning
    # for epoch in range(args.epoch):
    #     print("Starting epoch: ", epoch)
    #     train_loss = 0.0
    #     test_loss = 0.0
    #     train_acc = 0.0
    #     test_acc = 0.0
    #     iteration = 0
        
    #     if(epoch>=160):
    #         for group in optimizer.param_groups:
    #             group['lr'] *= 1e-1
    #     elif(epoch>=120):
    #         for group in optimizer.param_groups:
    #             group['lr'] *= 1e-1
    #     elif(epoch>=80):
    #         for group in optimizer.param_groups:
    #             group['lr'] *= 1e-1
        
    #     student.train()
    #     for input, label in train_dataloader_u:
    #         input, label = input.to(CUDA), label.to(CUDA)
    #         output = student(input)
    #         output_idx = torch.argmax(output, dim=1)
    #         train_acc += torch.sum(output_idx == label)/len(input)
            
    #         with torch.inference_mode():
    #             logits = teacher(input)
    #         logits = logits.detach().clone()
            
    #         soft_targets = torch.nn.functional.softmax(logits, dim=1)
    #         loss = criterion(torch.log_softmax(output, dim=1), soft_targets)
            
    #         train_loss += loss.item()
    #         optimizer.zero_grad()
    #         loss.backward()
    #         optimizer.step() 
    #         iteration += 1
            
    #     print("| Train Loss | Train Acc   | {a:.4f} | {b:.4f} |".format(a=train_loss/iteration, b=train_acc.item()/iteration))  
        
    #     student.eval()
    #     iteration = 0
    #     for input, label in test_dataloader:
    #         input, label = input.to(CUDA), label.to(CUDA)
    #         with torch.inference_mode():
    #             output = student(input)
    #         output = output.detach()
    #         output_idx = torch.argmax(output, dim=1)
    #         test_acc += torch.sum(output_idx == label)/len(input)
    #         loss = criterion2(output, label)
    #         test_loss += loss.item()
            
    #         iteration += 1
            
        max_test_acc = max(max_test_acc, test_acc.item()/iteration)
    #     print("| Test Loss  | Test Acc    | {a:.4f} | {b:.4f} |".format(a=test_loss/iteration, b=test_acc.item()/iteration))
        
        if(best_loss > test_loss):
            best_loss = test_loss
            torch.save(student.state_dict(), BEST_PATH)
        
        ckpt_state = {'epoch': epoch + 1, 'state_dict': student.state_dict(),
                      'optimizer': optimizer.state_dict(),'meta-net-state':meta_net.state_dict(),
                      'meta-optimizer': meta_optimizer.state_dict()}#,'meta-scheduler': meta_scheduler.state_dict()}
        #,'scheduler': scheduler.state_dict()
        
        torch.save(ckpt_state, LAST_PATH)
        
        print('Best Test Acc: ', max_test_acc,file=log_file)
        print("\n",file=log_file)
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--subset_size', '-s', type=str)
    parser.add_argument('--teacher_model', '-tm', type=str)
    parser.add_argument('--stu_model', '-sm', type=str)
    parser.add_argument('--data_set', '-d', type=str)
    parser.add_argument('--epoch', '-e', type=int, default=200)
    parser.add_argument('--temp', '-t', type=int, default=1)
    parser.add_argument('--cuda', '-c', type=str, default='0')
    parser.add_argument('--seed', '-se', type=int, default=0)
    parser.add_argument('--mcd_weight', '-w', type=float, default=0.01)
    parser.add_argument('--checkpoint', '-ch', type=bool, default=False)
    parser.add_argument('--meta_lr', '-mlr', type=float, default=1e-03)
    parser.add_argument('--meta_weight_decay', '-mwd', type=float, default=1e-04)
    
    
    args = parser.parse_args()
    main(args)