from PIL import Image
import os
import re

def remove_suffix(text):
    return re.sub(r'_\d+$', '', text)

folder = '/SSD/p76111262/CIC-IDS2018-ZSL/Auth/train'
fnames, labels = [], []
        
# Read all files within each label folder
for label in sorted(os.listdir(folder)):
    print("label:", label)
    label_name = remove_suffix(label)
    class_folder = os.path.join(folder, label)
    fnames.append(class_folder)
    labels.append(label_name)   

print("fnames:", fnames)
print("labels:", labels)
