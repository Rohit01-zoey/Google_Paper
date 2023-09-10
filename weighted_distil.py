import numpy as np
import random
import torch 
import os
import argparse
from models.model import MobileNetV1Depth2, MobileNetV1Depth1
from models.resnet import resnet110_cifar, resnet20_cifar, resnet56_cifar, resnet110_cifarv2, resnet56_cifarv2
import torchvision.transforms as transforms
from torchvision.datasets import CIFAR10, CIFAR100, SVHN, CelebA
from torch.utils.data import random_split, DataLoader
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsRegressor

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
    
    elif data == "celeb_a":
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

    else:
        raise NotImplementedError
    
    if data in ["cifar10","cifar100"]:
        subset_length = int(len(train_dataset) * subset_size)  # if subset_size < 1, it is a ratio, otherwise it is the number of samples
    else:
        subset_dict = {0.15 : 7500, 0.20 : 10000, 0.25:12500, 0.30 : 15000, 0.35 : 17500 }
        subset_length = int(subset_dict[subset_size])
    train_dataset_labelled, val_dataset, train_dataset_unlabelled = random_split(train_dataset, [subset_length, 2000, len(train_dataset) - subset_length-2000])

    train_dataloader_l = DataLoader(train_dataset_labelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    # train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=int((128 * (1.0-subset_size))/subset_size)+1, shuffle=True, num_workers=0, pin_memory=True)
    #train_dataloader_u = DataLoader(train_dataset_unlabelled, batch_size=128 , shuffle=True, num_workers=0, pin_memory=True)
    test_dataloader = DataLoader(test_dataset, batch_size=128, shuffle=True, num_workers=0, pin_memory=True)
    
    return train_dataloader_l, val_dataset, train_dataloader_u, test_dataloader, num_cls

# def confidence(softmaxes):
#     '''Implement the margin score for the given input'''
#     sorted_softmaxes, _ = torch.sort(softmaxes, dim = -1, descending=True)
#     return sorted_softmaxes[0, 0] - sorted_softmaxes[0, 1]

def margin(x: torch.Tensor,
           with_logits = True,
           normalize = False) -> torch.Tensor:
    """Computes the margin of a probability/logit tensor."""

    if with_logits:
        class_probabilities = F.softmax(x, dim=-1)
    else:
        class_probabilities = x
    a, _ = torch.topk(class_probabilities, k=2, dim=1)
    marg = (a[:, 0] - a[:, 1]).unsqueeze(1)

    if normalize:
        marg = marg / marg.mean()

    return marg

def entropy(x: torch.Tensor, with_logits=True, normalize=False) -> torch.Tensor:
    """Computes the entropy of a probability/logit tensor."""
    if with_logits:
        class_probabilities = F.softmax(x, dim=-1)
    else:
        class_probabilities = x

    ent = F.cross_entropy(class_probabilities, class_probabilities)
    if normalize:
        ent = ent / ent.mean()

    return ent

def KNNModel(teacher, student, n_neighbors, validation_dataset, uncertainty):
    cce = torch.nn.CrossEntropyLoss(reduction='none').to(CUDA)
    clip_max = 100000
    clip_min = 0.00001
    _val_loader = DataLoader(validation_dataset, batch_size=1, shuffle=True, num_workers=0, pin_memory=True)
    
    with torch.no_grad():
        student_predictions_validation = torch.cat([student(images.to(CUDA)) for images, _ in _val_loader])
        teacher_predictions_validation = torch.cat([teacher(images.to(CUDA)) for images, _ in _val_loader])
        
        teacher_uncertainty_validation = torch.reshape(
            uncertainty(teacher_predictions_validation), (-1, 1))

        student_uncertainty_validation = torch.reshape(
            uncertainty(student_predictions_validation), (-1, 1))
        
        noisy_loss = cce(student_predictions_validation, teacher_predictions_validation)
        clean_loss = cce(student_predictions_validation, torch.tensor([label for _, label in _val_loader]).to(CUDA))
        
        distortion_validation = noisy_loss / (clean_loss + clip_min)
        
        accuracy_validation = torch.eq(
            torch.argmax(teacher_predictions_validation, dim=1),
            torch.tensor([label for _, label in _val_loader]).to(CUDA)).float() # ch if labels are OHE
        # print(accuracy_validation.sum() / len(accuracy_validation))
        
        accuracy_validation = torch.reshape(accuracy_validation, shape=(-1, 1))
        distortion_validation = torch.reshape(distortion_validation, shape=(-1, 1))

        distortion_validation_masked = distortion_validation * (1. - accuracy_validation)
        
        distortion_validation = distortion_validation_masked + accuracy_validation
        
        distortion_validation_clipped = torch.clamp(distortion_validation, min=0., max=clip_max)
        
        space = torch.cat((teacher_uncertainty_validation, student_uncertainty_validation), dim=1).cpu()
        y = torch.cat((accuracy_validation, distortion_validation_clipped), dim=1).cpu()

        neigh = KNeighborsRegressor(n_neighbors=n_neighbors)
        neigh.fit(space, y)
        
        return neigh
    
def return_weights(teacher, student, KNNmodel, dataset, uncertainty):
    with torch.no_grad():
        teacher_predictions_train = teacher(dataset)# torch.cat([teacher(images.to(CUDA)) for images, _ in dataset])
        teacher_uncertainty_train = torch.reshape(uncertainty(teacher_predictions_train), shape=(-1, 1))

        student_train_predictions = student(dataset) # torch.cat([student(images.to(CUDA)) for images, _ in dataset])
        student_uncertainty_train = torch.reshape(uncertainty(student_train_predictions), shape=(-1, 1))
        
        uncertainty_train = torch.cat((teacher_uncertainty_train, student_uncertainty_train), dim=1)

        accuracy_distortion = KNNmodel.predict(uncertainty_train.cpu())

        accuracy_predicted = accuracy_distortion[:, :1]
        distortion_predicted = accuracy_distortion[:, 1:]
        # print((1. - accuracy_predicted) * (distortion_predicted - 1))
        
        weight = torch.tensor(1. / (1. + (1. - accuracy_predicted) * (distortion_predicted - 1) + 1e-8))
        weight = torch.clamp(weight, min=0., max=1.)
        weight = torch.reshape(weight, shape=(1, -1))
        
        return weight.to(CUDA)

# def make_dataset(teacher,student, V):
#     """_summary_

#     Args:
#         teacher (_type_): _description_
#         student (_type_): _description_
#         V (_type_): _description_
#     """
#     print("making the dataset...")
#     clip_max = 100000
#     clip_min = .00001
#     k = int(0.5*np.sqrt(len(V))) + 1 # size of the k-NN return qeury
#     criterion_KD = torch.nn.KLDivLoss(reduction='sum').to(CUDA)
#     criterion_CE = torch.nn.CrossEntropyLoss().to(CUDA)
#     U = [] # empty list
#     V_loader_ = DataLoader(V, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
#     for input, label in V_loader_:
#         input, label = input.to(CUDA), label.to(CUDA)
#         X = [confidence(teacher(input)), confidence(student(input))]
#         if torch.argmax(teacher(input), axis = -1)[0]+1 == label:
#             y = (0, 1)
#         else:
#             y = (1, criterion_KD(torch.log_softmax(student(input), dim = -1), torch.softmax(teacher(input), dim = -1)).item()/(criterion_CE(student(input), label).item()+1e-8))
#         print(y)
#         U.append((X, y))
#     print("...finished making the dataset")
#     return k, U # return the k value and the new dataset

# def weight_instance(input, label, k, U, teacher,student):
#     query = [confidence(teacher(input)), confidence(student(input))]
#     dist = []
#     for i in range(len(U)):
#         dist.append(torch.linalg.vector_norm(torch.tensor(U[i][0])-torch.tensor(query), ord = 2, dim = None))
#     _, indices = torch.topk(torch.tensor(dist), k, largest=False)
#     label_weight_instance_ = (1.0/k)*torch.sum(torch.stack([torch.tensor(U[i][1]) for i in indices], dim = -1), dim = -1)
#     p, d = label_weight_instance_[0], label_weight_instance_[1]
#     print("label_weight_instance_ = {}, p = {}, d = {}".format(label_weight_instance_, p, d))
#     return min(1, 1.0/(1 + p*(d-1)))


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
    
    return model

def main(args):
    global CUDA
    CUDA = "cuda:"+args.cuda
    set_seed(args.seed)
    train_dataloader_l, val_dataset, train_dataloader_u, test_dataloader, num_cls = data(float(args.subset_size),args.data_set)

    student = get_model(args.stu_model,num_cls)
    teacher = get_model(args.teacher_model,num_cls)
    teacher.load_state_dict(torch.load(\
        "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+'/best_model.pt',map_location='cuda:0'))
    teacher.eval()
    
    criterion_KD = torch.nn.KLDivLoss(reduction='none').to(CUDA)
    criterion_CE = torch.nn.CrossEntropyLoss().to(CUDA)
    
    optimizer = torch.optim.Adam(student.parameters(), lr=1e-3, weight_decay = 1e-4)
    #optimizer =torch.optim.SGD(student.parameters() , lr=0.1,momentum=0.9,weight_decay=5e-4, nesterov=True)
    #scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epoch//4)
    
    # k, U_data = make_dataset(teacher,student, val_dataset)
    # iter_loader = DataLoader(train_dataset_unlabelled, batch_size=1, shuffle=False, num_workers=0, pin_memory=True)
    # for input, label in iter_loader:
    #     input = input.to(CUDA)
    #     print(weight_instance(input, label, k, U_data, teacher,student))
    
    
    # The dimension of the covariate space of the KNN is d = 2.
    d = 2
    # We use the asymptotic formula to compute the number of neighbors.
    n_neigh = round(0.5 * (2000)**(2 / (2 + d))) #! change the validation size as need be

    
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
    
    DIR = "./results/"+args.data_set+"/"+args.subset_size+"/"+args.teacher_model+"_"+args.stu_model+"/weighted_distil"
    
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
        #! call the estimate_weight function to return the weights of the currently trained student model and pretrained teacher model
               
        #! call the weighted_knn function to return the weighted_knn model 
        knn = KNNModel(teacher, student, n_neigh, val_dataset, margin)
        
         
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
            
            #compute the weights for the unlabelled data
            weight = return_weights(teacher, student, knn, input_u, margin)
            
            soft_targets = torch.nn.functional.softmax(logits/Temp, dim=1)
            loss_u = torch.sum(criterion_KD(torch.log_softmax(output_u/Temp, dim=1), soft_targets),dim=1)
            train_loss_u += (Temp**2)*torch.mean(loss_u).item()
            
            # soft_targets = torch.nn.functional.softmax(logits/Temp, dim=1)
            # loss_u = (Temp**2)*criterion_KD(torch.log_softmax(output_u/Temp, dim=1), soft_targets)
            # train_loss_u += loss_u.item()
            
            # total_loss = loss + loss_u
            total_loss = loss + (Temp**2)*torch.mean(weight * loss_u )
            
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