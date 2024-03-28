import torch
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
import cv2

def print_examples(model, device, dataset):
    # transform = transforms.Compose(
    #     [
    #         transforms.Resize((299, 299)),
    #         transforms.ToTensor(),
    #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5)),
    #     ]
    # )

    model.eval()

    file_dir = "/SSD/ne6101157/pac4_mini/test/GoldenEye/GoldenEye_10"
    buffer = np.empty((512, 64, 64, 1), np.dtype('float32'))
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    for i, frame_name in enumerate(frames):
        img = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)
        frame = np.array(cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.float64)
        frame = frame.reshape(frame.shape[0], frame.shape[1], -1)
        buffer[i] = frame
    buffer = torch.tensor(buffer).unsqueeze(0)
    buffer = buffer.permute(0, 4, 1, 2, 3)
    
    print('Example 1 CORRECT:alert http $EXTERNAL_NET any -> $HTTP_SERVERS any (msg:"ET DOS Inbound GoldenEye DoS attack"; flow:established,to_server; threshold: type both, track by_src, count 100, seconds 300; http.uri; content:"/?"; fast_pattern; depth:2; content:"="; distance:3; within:11; pcre:"/^\/\?[a-zA-Z0-9]{3,10}=[a-zA-Z0-9]{3,20}(?:&[a-zA-Z0-9]{3,10}=[a-zA-Z0-9]{3,20})*?$/"; http.header; content:"Keep|2d|Alive|3a|"; content:"Connection|3a| keep|2d|alive"; content:"Cache|2d|Control|3a|"; pcre:"/^Cache-Control\x3a\x20(?:max-age=0|no-cache)\r?$/m"; content:"Accept|2d|Encoding|3a|"; classtype:denial-of-service; sid:2018208; rev:3; metadata:created_at 2014_03_05, updated_at 2020_04_28;)')
    print(
        "Example 1 OUTPUT: "
        + " ".join(model.caption_video(buffer.to(device), dataset.vocab))
    )
    file_dir = "/SSD/ne6101157/pac4_mini/test/LOIC/LOIC_10"
    buffer = np.empty((512, 64, 64, 1), np.dtype('float32'))
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    for i, frame_name in enumerate(frames):
        img = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)
        frame = np.array(cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.float64)
        frame = frame.reshape(frame.shape[0], frame.shape[1], -1)
        buffer[i] = frame
    buffer = torch.tensor(buffer).unsqueeze(0)
    buffer = buffer.permute(0, 4, 1, 2, 3)
    
    print('Example 2 CORRECT:alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:“ET DOS Inbound Low Orbit Ion Cannon LOIC DDOS Tool desu string”; flow:to_server,established; content:“desudesudesu”; nocase; threshold: type limit,track by_src,seconds 180,count 1; reference:url,www.isc.sans.org/diary.html?storyid=10051; classtype:trojan-activity; sid:2012049; rev:5; metadata:created_at 2010_12_13, updated_at 2010_12_13;)')
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_video(buffer.to(device), dataset.vocab))
    )
    file_dir = "/SSD/ne6101157/pac4_mini/test/LOIC/LOIC_10"
    buffer = np.empty((512, 64, 64, 1), np.dtype('float32'))
    frames = sorted([os.path.join(file_dir, img) for img in os.listdir(file_dir)])
    for i, frame_name in enumerate(frames):
        img = cv2.imread(frame_name, cv2.IMREAD_GRAYSCALE)
        frame = np.array(cv2.resize(img, (64, 64), interpolation=cv2.INTER_CUBIC)).astype(np.float64)
        frame = frame.reshape(frame.shape[0], frame.shape[1], -1)
        buffer[i] = frame
    buffer = torch.tensor(buffer).unsqueeze(0)
    buffer = buffer.permute(0, 4, 1, 2, 3)
    
    print('Example 2 CORRECT:alert tcp $EXTERNAL_NET any -> $HOME_NET any (msg:“ET DOS Inbound Low Orbit Ion Cannon LOIC DDOS Tool desu string”; flow:to_server,established; content:“desudesudesu”; nocase; threshold: type limit,track by_src,seconds 180,count 1; reference:url,www.isc.sans.org/diary.html?storyid=10051; classtype:trojan-activity; sid:2012049; rev:5; metadata:created_at 2010_12_13, updated_at 2010_12_13;)')
    print(
        "Example 2 OUTPUT: "
        + " ".join(model.caption_video(buffer.to(device), dataset.vocab))
    )
    # test_img4 = transform(
    #     Image.open("test_examples/boat.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 4 CORRECT: A small boat in the ocean")
    # print(
    #     "Example 4 OUTPUT: "
    #     + " ".join(model.caption_image(test_img4.to(device), dataset.vocab))
    # )
    # test_img5 = transform(
    #     Image.open("test_examples/horse.png").convert("RGB")
    # ).unsqueeze(0)
    # print("Example 5 CORRECT: A cowboy riding a horse in the desert")
    # print(
    #     "Example 5 OUTPUT: "
    #     + " ".join(model.caption_image(test_img5.to(device), dataset.vocab))
    # )
    # model.train()


def save_checkpoint(state, filename="my_checkpoint.pth.tar"):
    print("=> Saving checkpoint")
    torch.save(state, filename)


def load_checkpoint(checkpoint, model, optimizer):
    print("=> Loading checkpoint")
    model.load_state_dict(checkpoint["state_dict"])
    optimizer.load_state_dict(checkpoint["optimizer"])
    step = checkpoint["step"]
    return step

