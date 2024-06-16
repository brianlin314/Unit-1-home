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
    def __init__(self, split='train', dataset='CIC-IDS2018', clip_len=16, embedding_map=None, seen_list=None, unseen_list=None):
        self.split = split
        self.clip_len = clip_len
        self.resize_height = 64
        self.resize_width = 64
        self.embedding_map = embedding_map
        _, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        
        self.fnames, self.labels_index, self.labels_embedding = [], [], []
        seen_num = len(seen_list)
        for label in sorted(os.listdir(folder)):
            class_folder = os.path.join(folder, label)
            self.fnames.append(class_folder)
            # 判斷是不是 seen_list 中的類別
            label = remove_suffix(label)
            if label in seen_list:
                self.labels_index.append(seen_list.index(label))
            elif label not in seen_list and label in unseen_list:
                self.labels_index.append(unseen_list.index(label) + seen_num)  
            else:
                print(f"Unknown class {label}")
                raise ValueError(f"Unknown class {label}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))
        embedding = torch.from_numpy(self.embedding_map[self.labels_index[index]])
        idx = torch.tensor(self.labels_index[index], dtype=torch.long)

        return buffer, embedding, idx

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
        is_seen = 0 if parent_dir not in self.unseen_class else 1
        return image, is_seen
    
class VideoImageDataset(Dataset):
    def __init__(self, split='train', dataset='CIC-IDS2018', clip_len=16, embedding_map=None, seen_list=None, unseen_list=None):
        self.split = split
        self.clip_len = clip_len
        self.resize_height = 64
        self.resize_width = 64
        self.embedding_map = embedding_map
        
        _, self.output_dir = Path.db_dir(dataset)
        self.folder = os.path.join(self.output_dir, split)
        
        self.fnames, self.labels_index = [], []
        seen_num = len(seen_list)
        for label in sorted(os.listdir(self.folder)):
            class_folder = os.path.join(self.folder, label)
            self.fnames += [os.path.join(class_folder, f) for f in os.listdir(class_folder)]
            label_clean = remove_suffix(label)
            if label_clean in seen_list:
                self.labels_index += [seen_list.index(label_clean)] * len(os.listdir(class_folder))
            elif label_clean in unseen_list:
                self.labels_index += [unseen_list.index(label_clean) + seen_num] * len(os.listdir(class_folder))
            else:
                raise ValueError(f"Unknown class {label}")

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        video_path = self.fnames[index]
        buffer = self.load_frames(video_path)
        buffer = torch.from_numpy(buffer.transpose((3, 0, 1, 2)))  # Convert to tensor
        first_frame = buffer[0]  # 取第一帧作为图像数据
        embedding = torch.tensor(self.embedding_map[self.labels_index[index]], dtype=torch.float32)
        idx = torch.tensor(self.labels_index[index], dtype=torch.long)

        return first_frame, buffer, embedding, idx

    def load_frames(self, video_path):
        cap = cv2.VideoCapture(video_path)
        frames = []
        while True:
            ret, frame = cap.read()
            if not ret or len(frames) == self.clip_len:
                break
            frame = cv2.resize(frame, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC)
            frames.append(frame.astype(np.float32))
        cap.release()
        buffer = np.array(frames)
        return buffer



 