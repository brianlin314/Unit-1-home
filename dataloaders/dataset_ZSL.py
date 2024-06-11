import os
from sklearn.model_selection import train_test_split
from torchvision import transforms as t
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path
from PIL import Image
import re
import glob

def remove_suffix(text):
    return re.sub(r'_\d+$', '', text)

class VideoDataset(Dataset):
    def __init__(self, split='train', dataset='CIC-IDS2018', clip_len=16, embedding_map=None, attack_list=None):
        self.split = split
        self.clip_len = clip_len
        self.resize_height = 64
        self.resize_width = 64
        self.embedding_map = embedding_map

        _, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        
        self.fnames, self.labels_index, self.labels_embedding = [], [], []

        # Read all files within each label folder
        for label in sorted(os.listdir(folder)):
            label_name = remove_suffix(label)
            class_folder = os.path.join(folder, label)
            self.fnames.append(class_folder)
            self.labels_index.append(attack_list.index(label_name))   

        print(f"{split}_labels_index:", self.labels_index)
        # assert len(labels) == len(self.fnames)  # 確認每個file name都有自己的label
        label_file = os.path.join('dataloaders', dataset + '.txt')
        if not os.path.exists(label_file):
            with open(label_file, 'w') as f:
                for label in attack_list:
                    f.writelines(label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        idx = self.labels_index[index]
        embedding = self.embedding_map[idx]
        idx = np.array(idx)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(embedding), torch.from_numpy(idx)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        frame_count = len(frames)

        # 如果幀數少於clip_len，則重複列表中的元素
        if frame_count < self.clip_len:
            frames = (frames * ((self.clip_len // frame_count) + 1))[:self.clip_len]
        else:
            # 選取前clip_len個幀
            frames = frames[:self.clip_len]

        buffer = np.empty((self.clip_len, self.resize_height, self.resize_width, 3), np.dtype('float32'))

        # 讀取和處理每一幀
        for i, frame_name in enumerate(frames):
            img = cv2.imread(frame_name)
            if img is not None:  # 確保圖片讀取正常
                frame = cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC).astype(np.float32)
                buffer[i] = frame
            else:
                raise FileNotFoundError(f"Unable to read image {frame_name}. Ensure the file exists and is a valid image.")

        return buffer
    
class ImageDataset(Dataset):
    def __init__(self, root_dir, transform=None, unseen_class=None):
        self.root_dir = root_dir
        self.transform = transform
        self.unseen_class = unseen_class
        file_list = glob.glob(os.path.join(root_dir, '**', '0.png'), recursive=True)
        self.filenames = [os.path.abspath(file_path) for file_path in file_list]

    def __len__(self):
        return len(self.filenames)

    def __getitem__(self, idx):
        img_name = self.filenames[idx]
        image = Image.open(img_name)
        if self.transform:
            image = self.transform(image)

        parent_dir = os.path.basename(os.path.dirname(img_name))
        parent_dir = remove_suffix(parent_dir)
        is_seen = 0 if parent_dir in self.unseen_class else 1
        return image, is_seen