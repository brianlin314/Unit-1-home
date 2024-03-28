import os
from sklearn.model_selection import train_test_split

import torch
import cv2
import numpy as np
from torch.utils.data import Dataset
from mypath_pcap import Path


class VideoDataset(Dataset):
    r"""A Dataset for a folder of videos. Expects the directory structure to be
    directory->[train/val/test]->[class labels]->[videos]. Initializes with a list
    of all file names, along with an array of labels, with label being automatically
    inferred from the respective folder names.

        Args:
            dataset (str): Name of dataset. Defaults to 'ucf101'.
            split (str): Determines which folder of the directory the dataset will read from. Defaults to 'train'.
            clip_len (int): Determines how many frames are there in each clip. Defaults to 16.
            preprocess (bool): Determines whether to preprocess dataset. Default is False.
    """

    def __init__(self, dataset='pac4', split='train', clip_len=16, preprocess=True):
        self.root_dir, self.output_dir = Path.db_dir(dataset)
        folder = os.path.join(self.output_dir, split) #"/media/brian/Brian/2023/ready_pcap/train"
        self.clip_len = clip_len
        self.split = split

        # The following three parameters are chosen as described in the paper section 4.1
        self.resize_height = 128
        self.resize_width = 128
        self.crop_size = 128

        if not self.check_integrity(): # 檢查路徑是否存在
            raise RuntimeError('Dataset not found or corrupted.' +
                               ' You need to download it from official website.')

        if preprocess:  # check_preprocess()檢查前處理的資料是否正確 ，preprocess判斷是否已經前處理
            print('Preprocessing of {} dataset, this will take long, but it will be done only once.'.format(dataset))
            self.preprocess()

        # Obtain all the filenames of files inside all the class folders
        # Going through each class folder one at a time
        self.fnames, labels = [], []
        for label in sorted(os.listdir(folder)):                            #從切分好的train data中找所有資料夾名稱作為label
            for fname in os.listdir(os.path.join(folder, label)):           #把每個裝有一個影片的圖像檔案夾放進去fnames中 (ex: fnames = SSH_1, SSH_2, ...)
                self.fnames.append(os.path.join(folder, label, fname))
                labels.append(label)

        assert len(labels) == len(self.fnames)  #每個file name都有自己的label
        print('Number of {} videos: {:d}'.format(split, len(self.fnames)))

        # Prepare a mapping between the label names (strings) and indices (ints)
        self.label2index = {label: index for index, label in enumerate(sorted(set(labels)))}    #每個label有自己的index 用dictionary的方式儲存 （ex:{'SSH':1, 'goldeneye':2, ...}）
        # Convert the list of label names into an array of label indices
        self.label_array = np.array([self.label2index[label] for label in labels], dtype=int)   #每個label有自己的index 用array的方式儲存 （ex:['SSH', 'goldeneye', ...]）

        if dataset == "ucf101":
            if not os.path.exists('dataloaders/ucf_new.txt'):
                with open('dataloaders/ucf_new.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'hmdb51':
            if not os.path.exists('dataloaders/hmdb_labels.txt'):
                with open('dataloaders/hmdb_labels.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')

        elif dataset == 'pac4':  #創建label的txt檔
            if not os.path.exists('dataloaders/pac4.txt'):
                with open('dataloaders/pac4.txt', 'w') as f:
                    for id, label in enumerate(sorted(self.label2index)):
                        f.writelines(str(id+1) + ' ' + label + '\n')


    def __len__(self):
        return len(self.fnames)

    def __getitem__(self, index):
        # Loading and preprocessing.
        buffer = self.load_frames(self.fnames[index])
        buffer = self.crop(buffer, self.clip_len, self.crop_size)
        labels = np.array(self.label_array[index])

        buffer = self.normalize(buffer)
        buffer = self.to_tensor(buffer)
        return torch.from_numpy(buffer), torch.from_numpy(labels)

    def check_integrity(self):
        if not os.path.exists(self.root_dir):
            return False
        else:
            return True

    # def check_preprocess(self):
    #     # TODO: Check image size in output_dir
    #     if not os.path.exists(self.output_dir):
    #         return False
    #     elif not os.path.exists(os.path.join(self.output_dir, 'train')):
    #         return False

    #     for ii, video_class in enumerate(os.listdir(os.path.join(self.output_dir, 'train'))):
    #         for video in os.listdir(os.path.join(self.output_dir, 'train', video_class)):
    #             video_name = os.path.join(os.path.join(self.output_dir, 'train', video_class, video),
    #                                 sorted(os.listdir(os.path.join(self.output_dir, 'train', video_class, video)))[0])
    #             image = cv2.imread(video_name)
    #             if np.shape(image)[0] != 128 or np.shape(image)[1] != 171:
    #                 return False
    #             else:
    #                 break

    #         if ii == 10:
    #             break

    #     return True

    def preprocess(self):
        if not os.path.exists(self.output_dir):
            os.mkdir(self.output_dir)
            os.mkdir(os.path.join(self.output_dir, 'train'))
            os.mkdir(os.path.join(self.output_dir, 'val'))
            os.mkdir(os.path.join(self.output_dir, 'test'))

        # Split train/val/test sets
        for file in os.listdir(self.root_dir):
            file_path = os.path.join(self.root_dir, file)               #ex:root_dir = (/media/brian/Brian/2023/splited_pcap/), file = SSH
            print('file_path: ',file_path)
            video_files = [name for name in os.listdir(file_path) if os.path.isdir(os.path.join(file_path, name))]      #ex: video_file = [SSH_1, SSH2, ...](all attack folder)

            train_and_valid, test = train_test_split(video_files, test_size=0.2, random_state=42)
            train, val = train_test_split(train_and_valid, test_size=0.2, random_state=42)          #train:0.64, val:0.16, test:0.2

            print('train', train)
            print('val', val)
            print('test', test)

            train_dir = os.path.join(self.output_dir, 'train', file)
            val_dir = os.path.join(self.output_dir, 'val', file)
            test_dir = os.path.join(self.output_dir, 'test', file)

            if not os.path.exists(train_dir):
                os.makedirs(train_dir)
            if not os.path.exists(val_dir):
                os.makedirs(val_dir)
            if not os.path.exists(test_dir):
                os.makedirs(test_dir)

            for att_folder in train:
                command = 'cp -r '+ file_path + "/" + att_folder + " " + train_dir
                os.system(command)

            for att_folder in val:
                command = 'cp -r '+ file_path + "/" + att_folder + " " + val_dir
                os.system(command)

            for att_folder in test:
                command = 'cp -r '+ file_path + "/" + att_folder + " " + test_dir
                os.system(command)

        print('Preprocessing finished.')

    def randomflip(self, buffer):
        """Horizontally flip the given image and ground truth randomly with a probability of 0.5."""

        if np.random.random() < 0.5:
            for i, frame in enumerate(buffer):
                frame = cv2.flip(buffer[i], flipCode=1)
                buffer[i] = cv2.flip(frame, flipCode=1)

        return buffer


    def normalize(self, buffer):
        for i, frame in enumerate(buffer):
            frame -= np.array([[[90.0, 98.0, 102.0]]])
            buffer[i] = frame
#ex:
        return buffer

    def to_tensor(self, buffer):
        return buffer.transpose((3, 0, 1, 2))

    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 3), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            frame = np.array(cv2.imread(frame_name)).astype(np.float64)
            buffer[i] = frame

        return buffer

    def crop(self, buffer, clip_len, crop_size):
        # randomly select time index for temporal jittering
        time_index = np.random.randint(buffer.shape[0] - clip_len)

        # Randomly select start indices in order to crop the video
        height_index = np.random.randint(buffer.shape[1] - crop_size)
        width_index = np.random.randint(buffer.shape[2] - crop_size)

        # Crop and jitter the video using indexing. The spatial crop is performed on
        # the entire array, so each frame is cropped in the same location. The temporal
        # jitter takes place via the selection of consecutive frames
        buffer = buffer[time_index:time_index + clip_len,
                 height_index:height_index + crop_size,
                 width_index:width_index + crop_size, :]

        return buffer





if __name__ == "__main__":
    from torch.utils.data import DataLoader
    train_data = VideoDataset(dataset='pac4', split='train', clip_len=32, preprocess=True)
    train_loader = DataLoader(train_data, batch_size=100, shuffle=True, num_workers=4)

    for i, sample in enumerate(train_loader):
        inputs = sample[0]
        labels = sample[1]
        print(inputs.size())
        print(labels)

        if i == 1:
            break
