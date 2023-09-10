
#CUDA_VISIBLE_DEVICES=1 python3 main.py -s 0.15 -d cifar100 -m ResNet56

#python3 distil.py -s 0.15 -d cifar100 -sm ResNet20 -tm ResNet110

# python3 meta_distil_corr.py -s 0.15 -d cifar100 -sm ResNet20 -tm ResNet110

#################################################################################################################################################################################

# all the changes made by me are beyond this point == please delete all the lines of the code below this comment line
# arigatou gozaimasu

#!/bin/bash
# python3 main.py -s 0.15 -m MobileNetV1Depth2 -d cifar10 & python3 main.py -s 0.20 -m MobileNetV1Depth2 -d cifar10 & python3 main.py -s 0.25 -m MobileNetV1Depth2 -d cifar10 & python3 main.py -s 0.30 -m MobileNetV1Depth2 -d cifar10 & python3 main.py -s 0.35 -m MobileNetV1Depth2 -d cifar10
# if [ $? -eq 0 ]; then       
#     python3 main.py -s 0.15 -m MobileNetV1Depth1 -d cifar10 &  python3 main.py -s 0.20 -m MobileNetV1Depth1 -d cifar10 & python3 main.py -s 0.25 -m MobileNetV1Depth1 -d cifar10 & python3 main.py -s 0.30 -m MobileNetV1Depth1 -d cifar10 &   python3 main.py -s 0.35 -m MobileNetV1Depth1 -d cifar10
# else
#     echo "Failed"
# fi

# pure distillation temp = 1 and different subset sizes
# python3 distil.py -s 0.15 -sm MobileNetV1Depth1 -tm MobileNetV1Depth2 -d cifar10  & python3 distil.py -s 0.20 -sm MobileNetV1Depth1 -tm MobileNetV1Depth2 -d cifar10  & python3 distil.py -s 0.25 -sm MobileNetV1Depth1 -tm MobileNetV1Depth2 -d cifar10  & python3 distil.py -s 0.30 -sm MobileNetV1Depth1 -tm MobileNetV1Depth2 -d cifar10  & python3 distil.py -s 0.35 -sm MobileNetV1Depth1 -tm MobileNetV1Depth2 -d cifar10 

# meta distillation temp = 1 and different subset sizes
# python3 meta_distil_corr.py -s 0.15 -tm MobileNetV1Depth2 -sm MobileNetV1Depth1 -d cifar10 & python3 meta_distil_corr.py -s 0.20 -tm MobileNetV1Depth2 -sm MobileNetV1Depth1 -d cifar10 
# if [ $? -eq 0 ]; then  
#      python3 meta_distil_corr.py -s 0.25 -tm MobileNetV1Depth2 -sm MobileNetV1Depth1 -d cifar10 & python3 meta_distil_corr.py -s 0.30 -tm MobileNetV1Depth2 -sm MobileNetV1Depth1 -d cifar10 & python3 meta_distil_corr.py -s 0.35 -tm MobileNetV1Depth2 -sm MobileNetV1Depth1 -d cifar10
# else
#     echo "Failed"
# fi

# using the cifar100 dataset and the same subset sizes with temperature = 1

python3 main.py -s 0.15 -m ResNet110 -d cifar100 & python3 main.py -s 0.20 -m ResNet110 -d cifar100 & python3 main.py -s 0.25 -m ResNet110 -d cifar100 & python3 main.py -s 0.30 -m ResNet110 -d cifar100 & python3 main.py -s 0.35 -m ResNet110 -d cifar100
   
python3 main.py -s 0.15 -m ResNet20 -d cifar100 &  python3 main.py -s 0.20 -m ResNet20 -d cifar100 & python3 main.py -s 0.25 -m ResNet20 -d cifar100 & python3 main.py -s 0.30 -m ResNet20 -d cifar100 &   python3 main.py -s 0.35 -m ResNet20 -d cifar100

# running the distillation step for temperature = 1
python3 distil.py -s 0.15 -sm ResNet20 -tm ResNet110 -d cifar100 & python3 distil.py -s 0.20 -sm ResNet20 -tm ResNet110 -d cifar100 & python3 distil.py -s 0.25 -sm ResNet20 -tm ResNet110 -d cifar100 & python3 distil.py -s 0.30 -sm ResNet20 -tm ResNet110 -d cifar100 & python3 distil.py -s 0.35 -sm ResNet20 -tm ResNet110 -d cifar100 

# running the meta-disticllation step for temperature = 1
python3 meta_distil_corr.py -s 0.15 -tm ResNet110 -sm ResNet20 -d cifar100 & python3 meta_distil_corr.py -s 0.20 -tm ResNet110 -sm ResNet20 -d cifar100 
if [ $? -eq 0 ]; then  
     python3 meta_distil_corr.py -s 0.25 -tm ResNet110 -sm ResNet20 -d cifar100 & python3 meta_distil_corr.py -s 0.30 -tm ResNet110 -sm ResNet20 -d cifar100 & python3 meta_distil_corr.py -s 0.35 -tm ResNet110 -sm ResNet20 -d cifar100
else
    echo "Failed"
fi