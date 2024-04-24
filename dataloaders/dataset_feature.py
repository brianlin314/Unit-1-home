import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset

class VideoDataset(Dataset):

    def __init__(self, dataset='CIC-IDS2018', clip_len=16):
        folder = dataset
        self.clip_len = clip_len
        self.resize_height = 64
        self.resize_width = 64
        self.fnames = []
        for fname in os.listdir(folder):    
            self.fnames.append(os.path.join(folder, fname))
        # print("fnames:", self.fnames)
        

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        file_num = self.fnames[index].split('/')[-1].split('.')[0]
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), file_num

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        # print("file_dir:", file_dir, " frames:", frames)
        frame_count = len(frames)
        frames_to_load = frames[:self.clip_len] if frame_count >= self.clip_len else frames
        buffer = np.empty((len(frames_to_load), self.resize_height, self.resize_width, 3), np.dtype('float32')) 
        for i, frame_name in enumerate(frames):
            img = cv2.imread(frame_name)  
            frame = np.array(cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC)).astype(np.float32)
            buffer[i] = frame

        return buffer
