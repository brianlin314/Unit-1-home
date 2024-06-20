import os
import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath import Path

class VideoDataset(Dataset):
    def __init__(self, split='train', dataset='CIC-IDS2018', clip_len=16):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split)
        self.split = split

        self.clip_len = clip_len
        self.resize_height = 64
        self.resize_width = 64

        if not self.check_integrity():     # 檢查路徑是否存在
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):     # 從切分好的train data中找所有資料夾名稱作為label
            for fname in os.listdir(os.path.join(folder, label)):     # 把每個裝有一個影片的圖像檔案夾放進去fnames中 (ex: fnames = SSH_1, SSH_2, ...)
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)  # 每個file name都有自己的label
        # print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}     # 每個label有自己的index 用dictionary的方式儲存 （ex:{'SSH':1, 'goldeneye':2, ...}）
        # print("label2index:", self.label2index)
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)      # 每個label有自己的index 用array的方式儲存 （ex:['SSH', 'goldeneye', ...]）

        label_file = os.path.join('dataloaders', dataset + '.txt')
        if not os.path.exists(label_file):
            with open(label_file, 'w') as f:
                for label in sorted(self.label2index):
                    f.writelines(label + '\n')

    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        buffer = self.load_frames(self.fnames[index])
        labels = np.array(self.label_array[index])
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)], key=lambda x: int(x.split('/')[-1].split('.')[0]))
        frame_count = len(frames)

        frames_to_load = frames[:self.clip_len] if frame_count >= self.clip_len else frames
        buffer = np.empty((len(frames_to_load), self.resize_height, self.resize_width, 3), np.dtype('float32')) 
        for i, frame_name in enumerate(frames):
            img = cv2.imread(frame_name)  
            frame = np.array(cv2.resize(img, (self.resize_width, self.resize_height), interpolation=cv2.INTER_CUBIC)).astype(np.float32)
            buffer[i] = frame

        return buffer
