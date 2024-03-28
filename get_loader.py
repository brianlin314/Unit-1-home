import os  # when loading file paths
import pandas as pd  # for lookup in annotation file
import spacy  # for tokenizer
import torch
from torch.nn.utils.rnn import pad_sequence  # pad batch
from torch.utils.data import DataLoader, Dataset
from PIL import Image  # Load img
import torchvision.transforms as transforms
import numpy as np
import cv2

# We want to convert text -> numerical values
# 1. We need a Vocabulary mapping each word to a index
# 2. We need to setup a Pytorch dataset to load the data
# 3. Setup padding of every batch (all examples should be
#    of same seq_len and setup dataloader)
# Note that loading the image is very easy compared to the text!

# Download with: python -m spacy download en
spacy_eng = spacy.load("en_core_web_sm")


class Vocabulary:
    def __init__(self, freq_threshold):
        self.itos = {0: "<PAD>", 1: "<SOS>", 2: "<EOS>", 3: "<UNK>"}
        self.stoi = {"<PAD>": 0, "<SOS>": 1, "<EOS>": 2, "<UNK>": 3}
        self.freq_threshold = freq_threshold

    def __len__(self):
        return len(self.itos)

    @staticmethod
    def tokenizer_eng(text):
        if isinstance(text, tuple) == True:
            text = list(text)[0]
        return [tok.text.lower() for tok in spacy_eng.tokenizer(text)]

    def build_vocabulary(self, sentence_list):
        frequencies = {}
        idx = 4
        for sentence in sentence_list:
            for word in self.tokenizer_eng(sentence):
                if word not in frequencies:
                    frequencies[word] = 1

                else:
                    frequencies[word] += 1

                if frequencies[word] == self.freq_threshold:
                    self.stoi[word] = idx
                    self.itos[idx] = word
                    idx += 1

    def numericalize(self, text):
        tokenized_text = self.tokenizer_eng(text)

        return [
            self.stoi[token] if token in self.stoi else self.stoi["<UNK>"]
            for token in tokenized_text
        ]


class PacDataset(Dataset):
    def __init__(self, root_dir="/SSD/ne6101157/pac4_mini_nonsplit", transform=None, freq_threshold=5):
        self.root_dir = root_dir
        print("root:",self.root_dir)

        self.transform = transform
        self.resize_height = 64
        self.resize_width = 64

        self.flow, rules = [], []
        for label in sorted(os.listdir(root_dir)):                            #從切分好的train data中找所有資料夾名稱作為label
            for fname in os.listdir(os.path.join(root_dir, label)):           #把每個裝有一個影片的圖像檔案夾放進去fnames中 (ex: fnames = SSH_1, SSH_2, ...)
                self.flow.append(os.path.join(root_dir, label, fname))
                if label == "GoldenEye":
                    rule = 'alert http $EXTERNAL_NET any -> $HTTP_SERVERS any (msg:"ET DOS Inbound GoldenEye DoS attack"; flow:established,to_server; threshold: type both, track by_src, count 100, seconds 300; http.uri; content:"/?"; fast_pattern; depth:2; content:"="; distance:3; within:11; pcre:"/^\/\?[a-zA-Z0-9]{3,10}=[a-zA-Z0-9]{3,20}(?:&[a-zA-Z0-9]{3,10}=[a-zA-Z0-9]{3,20})*?$/"; http.header; content:"Keep|2d|Alive|3a|"; content:"Connection|3a| keep|2d|alive"; content:"Cache|2d|Control|3a|"; pcre:"/^Cache-Control\x3a\x20(?:max-age=0|no-cache)\r?$/m"; content:"Accept|2d|Encoding|3a|"; classtype:denial-of-service; sid:2018208; rev:3; metadata:created_at 2014_03_05, updated_at 2020_04_28;)'
                elif label == "Heartbleed":
                    rule = 'alert tcp any any -> $HOME_NET !$HTTP_PORTS (msg:"ET EXPLOIT Malformed HeartBeat Request"; flow:established,to_server; content:"|18 03|"; depth:2; byte_test:1,<,4,2; content:"|01|"; offset:5; depth:1; byte_extract:2,3,record_len; byte_test:2,>,2,3; byte_test:2,>,record_len,6; threshold:type limit,track by_src,count 1,seconds 120; flowbits:set,ET.MalformedTLSHB; classtype:bad-unknown; sid:2018372; rev:3; metadata:created_at 2014_04_08, former_category CURRENT_EVENTS, updated_at 2014_04_08;)',
                elif label == "LOIC":
                    rule = 'alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:“ET DOS Inbound Low Orbit Ion Cannon LOIC DDOS Tool desu string”; flow:to_server,established; content:“desudesudesu”; nocase; threshold: type limit,track by_src,seconds 180,count 1; classtype:trojan-activity; sid:2012049; rev:5; metadata:created_at 2010_12_13, updated_at 2010_12_13;)";'
                elif label == "SSHPatator":
                    rule = 'alert tcp $EXTERNAL_NET any -> $HOME_NET 22 (msg:"ET SCAN Potential SSH Scan"; flow:to_server; flags:S,12; threshold: type both, track by_src, count 5, seconds 120; classtype:attempted-recon; sid:2001219; rev:20; metadata:created_at 2010_07_30, updated_at 2010_07_30;)'
                rules.append(rule)
        assert len(rules) == len(self.flow)  #每個file name都有自己的label
        print('Number of videos: {:d}'.format(len(self.flow)))
        # Get img, caption columns
        self.captions = rules
        print('Number of rules: {:d}'.format(len(self.captions)))
        # Initialize vocabulary and build vocab
        self.vocab = Vocabulary(freq_threshold)
        self.vocab.build_vocabulary(self.captions)

    def __len__(self):
        return len(self.flow)

    def __getitem__(self, index):
        caption = self.captions[index]
        flow_fname= self.flow[index]
        buffer = self.load_frames(flow_fname)
        # buffer = self.to_tensor(buffer) 

        numericalized_caption = [self.vocab.stoi["<SOS>"]]
        numericalized_caption += self.vocab.numericalize(caption)
        numericalized_caption.append(self.vocab.stoi["<EOS>"])

        return torch.tensor(buffer), torch.tensor(numericalized_caption)
    
    def load_frames(self, file_dir):
        frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
        frame_count = len(frames)
        # print("from ", file_dir, "import ", frame_count, "frames")
        buffer = np.empty((frame_count, self.resize_height, self.resize_width, 1), np.dtype('float32'))
        for i, frame_name in enumerate(frames):
            img = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)
            frame = np.array(cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.float64)
            frame = frame.reshape(frame.shape[0], frame.shape[1], -1)
            buffer[i] = frame

        return buffer
    
    # def to_tensor(self, buffer):
    #     return buffer.transpose((3, 0, 1, 2))

class MyCollate:
    def __init__(self, pad_idx):
        self.pad_idx = pad_idx

    def __call__(self, batch):
        flows = [item[0].unsqueeze(0) for item in batch] #batch裡是flow/rule unsquueze是為了多一個dim給batch size用
        flows = torch.cat(flows, dim=0)
        targets = [item[1] for item in batch]
        targets = pad_sequence(targets, batch_first=False, padding_value=self.pad_idx)

        return flows, targets


def get_loader(
    root_folder,
    transform,
    batch_size=4,
    num_workers=8,
    shuffle=True,
    pin_memory=True,
):
    dataset = PacDataset(root_folder, transform=transform)

    pad_idx = dataset.vocab.stoi["<PAD>"]

    loader = DataLoader(
        dataset=dataset,
        batch_size=batch_size,
        num_workers=num_workers,
        shuffle=shuffle,
        pin_memory=pin_memory,
        collate_fn=MyCollate(pad_idx=pad_idx),
    )

    return loader, dataset


if __name__ == "__main__":
    # transform = transforms.Compose(
    #     [transforms.Resize((224, 224)), transforms.ToTensor(),]
    # )

    loader, dataset = get_loader(
        "/SSD/ne6101157/pac4_mini_nonsplit", transform=None
    )

    for idx, (flows, captions) in enumerate(loader):
        print(flows.shape)
        print(captions.shape)
        break
