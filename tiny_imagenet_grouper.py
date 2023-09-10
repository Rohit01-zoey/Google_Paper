mapping = {}
with open("/home/hp/data/tiny-imagenet-200/val/val_annotations.txt", "r") as file:
    for line in file:
        image_name, class_name, *_ = line.split()
        mapping[image_name] = class_name
        
print(mapping['val_9955.JPEG'])

import os
import shutil

source_folder = "/home/hp/data/tiny-imagenet-200/val/images"
destination_folder = "/home/hp/data/tiny-imagenet-200/val_grouped"

if not os.path.exists(destination_folder):
    os.makedirs(destination_folder)

for image_name, class_name in mapping.items():
    class_folder = os.path.join(destination_folder, class_name)
    if not os.path.exists(class_folder):
        os.makedirs(class_folder)
    source_path = os.path.join(source_folder, image_name)
    destination_path = os.path.join(class_folder, image_name)
    shutil.copy(source_path, destination_path)
