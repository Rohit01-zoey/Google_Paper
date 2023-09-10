import numpy as np
import random
import torch 
import os
import argparse
from models.model import MobileNetV1Depth2, MobileNetV1Depth1
from models.resnet import resnet110_cifar, resnet20_cifar, resnet56_cifar, ResNet18, ResNet50
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, CelebA, ImageFolder
from torch.utils.data import random_split, DataLoader

CIFAR10_MEAN = (0.4914, 0.4822, 0.4465)
CIFAR10_STD = (0.2471, 0.2435, 0.2616)

def data(subset_size,data_set):
    
    torch.cuda.manual_seed(42)
    torch.manual_seed(42)
    
    if data_set == "cifar10":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])

        test_transform = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize((0.4914, 0.4822, 0.4465), (0.2023, 0.1994, 0.2010)),
        ])
        
        train_dataset = CIFAR10(root='../data/', train=True, transform=train_transform ,download=True)
        test_dataset = CIFAR10(root='../data/', train=False, transform=test_transform, download=True)

        num_cls =10

    elif data_set == "cifar100":
        
        crop_size=32
        train_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.RandomCrop(crop_size, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
        ])

        test_transform = transforms.Compose([
            transforms.Resize(crop_size),
            transforms.ToTensor(),
            transforms.Normalize((0.5071, 0.4865, 0.4409), (0.2673, 0.2564, 0.2762)),
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
        train_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize(crop_size),  transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        test_transform = transforms.Compose([transforms.RandomHorizontalFlip(), transforms.Resize(crop_size),  transforms.ToTensor(), transforms.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))])
        # train_dataset = ImageNet(root='../data/', split = 'train', transform=train_transform )
        # test_dataset = ImageNet(root='../data/', split = 'val', transform=test_transform)
        # Load the Tiny ImageNet training set
        train_dataset = ImageFolder('../data/tiny-imagenet-200/train', transform=train_transform)

        # Load the Tiny ImageNet validation set
        val_dataset = ImageFolder('../data/tiny-imagenet-200/val', transform=train_transform)

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
        subset_dict = {0.05 : 5000, 0.10 : 10000, 0.90:90000}
        subset_length = int(subset_dict[subset_size])
    else:
        raise NotImplementedError
    
    if data_set in ["cifar10","cifar100","svhn","celeb_a"]:
        train_dataset_labelled, train_dataset_unlabelled = random_split(train_dataset, [subset_length, len(train_dataset) - subset_length])
    elif data_set in ["tiny-imagenet"]:
        train_dataset_labelled, train_dataset_unlabelled = random_split(train_dataset, [subset_length, len(train_dataset) - subset_length])
    else:
        raise NotImplementedError
    
    train_dataloader_l = DataLoader(train_dataset_labelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=int((128 * (1.0-subset_size))/subset_size)+1, shuffle=True, num_workers=0, pin_memory=True)
    #train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=128 , shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_dataloader_l, train_dataloader_u, test_dataloader,num_cls

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
        model = resnet110_cifar(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet20":
        model = resnet20_cifar(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet56":
        model = resnet56_cifar(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet18":
        model = ResNet18(num_classes=num_cls).to(CUDA)
    elif model_name == "ResNet50":
        model = ResNet50(num_classes=num_cls).to(CUDA)
    
    return model

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda
    set_seed(args.seed)
    train_dataloader_l, train_dataloader_u, test_dataloader,num_cls = data(float(args.subset_size),args.data_set)

    student = get_model(args.stu_model,num_cls)
    teacher = get_model(args.teacher_model,num_cls)
    teacher.load_state_dict(torch.load(\
        "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+'/best_model.pt',map_location='cuda:0'))
    teacher.eval()
    
    criterion_KD = torch.nn.KLDivLoss(reduction='batchmean').to(CUDA)
    criterion_CE = torch.nn.CrossEntropyLoss().to(CUDA)
    
    if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
        optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay = 1e-2)
    elif args.data_set in ['tiny-imagenet']:
        optimizer =torch.optim.SGD(student.parameters() , lr=0.1,momentum=0.9,weight_decay=5e-4, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epoch//4)
    
    iteration = 0
    test_acc = 0.
    test_loss = 0.0
    for input, label in test_dataloader:
        input, label = input.to(CUDA), label.to(CUDA)
        with torch.inference_mode():
            output = teacher(input)
        output = output.detach()
        output_idx = torch.argmax(output, dim=1)
        test_acc += torch.sum(output_idx == label)/len(input)
        loss = criterion_CE(output, label)
        test_loss += loss.item()
        
        iteration += 1
    
    DIR = "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+"_"+args.stu_model+"/plain_distil"
    
    os.makedirs(DIR, exist_ok = True)

    BEST_PATH = DIR+"/best_model.pt"
    LAST_PATH = DIR+"/last_model.pt"
    log_file = open(DIR+"/output.txt", 'w')

    print("|Teacher Test Acc | {b:.4f} |".format(b=test_acc.item()/iteration),file=log_file)
    
    if args.checkpoint:
        checkpoint = torch.load(LAST_PATH)
        student.load_state_dict(checkpoint['state_dict'],map_location='cuda:0')
        epoch_ = checkpoint['epoch']
        optimizer.load_state_dict(checkpoint['optimizer'])
        #scheduler.load_state_dict(checkpoint['scheduler'])
    else:
        epoch_ = 0
    max_test_acc = 0.0

    Temp = args.temp
    
    # pre training
    for epoch in range(epoch_,args.epoch):
        print("Starting epoch: ", epoch, flush=True,file=log_file)
        train_loss = 0.0
        train_loss_u = 0.0
        test_loss = 0.0
        train_acc = 0.0
        train_acc_u = 0.0
        total_loss = 0.0
        test_acc = 0.0
        iteration = 0
        best_loss = float('inf')
        
        if args.data_set in ['cifar10', 'cifar100', 'svhn', 'celeb_a']:
            if(epoch>=180):
                for group in optimizer.param_groups:
                    group['lr'] = 0.5*1e-6
            elif(epoch>=160):
                for group in optimizer.param_groups:
                    group['lr'] = 1e-6
            elif(epoch>=120):
                for group in optimizer.param_groups:
                    group['lr'] = 1e-5
            elif(epoch>=80):
                for group in optimizer.param_groups:
                    group['lr'] = 1e-4
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
        #for (input, label) in train_dataloader_l:
            # Labelled
            input, label = input.to(CUDA), label.to(CUDA)
            output = student(input)
            output_idx = torch.argmax(output, dim=1)
            train_acc += torch.sum(output_idx == label)/len(input)
            loss = criterion_CE(output, label)
            train_loss += loss.item()
            '''optimizer.zero_grad()
            loss.backward()
            optimizer.step()'''
            
        #for (input_u, label_u) in  train_dataloader_u:
            # Unlabelled
            input_u, label_u = input_u.to(CUDA), label_u.to(CUDA)
            output_u = student(input_u)
            output_idx_u = torch.argmax(output_u, dim=1)
            train_acc_u += torch.sum(output_idx_u == label_u)/len(input_u)
            with torch.inference_mode():
                logits = teacher(input_u)
            logits = logits.detach().clone()
            
            soft_targets = torch.nn.functional.softmax(logits/Temp, dim=1)
            loss_u = (Temp**2)*criterion_KD(torch.log_softmax(output_u/Temp, dim=1), soft_targets)
            train_loss_u += loss_u.item()
            
            total_loss = loss + loss_u
            
            optimizer.zero_grad()
            total_loss.backward()
            optimizer.step() 
            #scheduler.step(epoch + iteration/len(train_dataloader_u))
            iteration += 1
        print("| Labelled   Loss | Labelled   Acc | {a:.4f} | {b:.4f} |".format(a=train_loss/iteration, b=train_acc.item()/iteration), flush=True,file=log_file)  
        print("| Unlabelled Loss | Unlabelled Acc | {a:.4f} | {b:.4f} |".format(a=train_loss_u/iteration, b=train_acc_u.item()/iteration), flush=True,file=log_file)
        print("| Total      Loss | Total      Acc | {a:.4f} | {b:.4f} |".format(a=total_loss/iteration, b=(train_acc * float(args.subset_size) + train_acc_u * (1.0 - float(args.subset_size)))/iteration), flush=True,file=log_file)
        
        student.eval()
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
    #         loss = criterion_KD(torch.log_softmax(output, dim=1), soft_targets)
            
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
    #         loss = criterion_CE(output, label)
    #         test_loss += loss.item()
            
    #         iteration += 1
            
        max_test_acc = max(max_test_acc, test_acc.item()/iteration)
    #     print("| Test Loss  | Test Acc    | {a:.4f} | {b:.4f} |".format(a=test_loss/iteration, b=test_acc.item()/iteration))
        
        if(best_loss > test_loss):
            best_loss = test_loss
            torch.save(student.state_dict(), BEST_PATH)
        
        ckpt_state = {'epoch': epoch + 1, 'state_dict': student.state_dict(),
                      'optimizer': optimizer.state_dict()}#,'scheduler': scheduler.state_dict()}
        
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
    parser.add_argument('--weight', '-w', type=float, default=1)
    parser.add_argument('--checkpoint', '-ch', type=bool, default=False)
    
    args = parser.parse_args()
    main(args)