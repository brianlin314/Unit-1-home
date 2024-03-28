import torch
import torch.nn as nn
import statistics
import torchvision.models as models
from network import R3D_model, R2Plus1D_model, C3D_model, HM3D_model
import tarfile

class EncoderCNN(nn.Module):
    def __init__(self, classifier, embed_size, pretrain=True):
        super(EncoderCNN, self).__init__()
        if classifier == "R3D":
            self.pretrain = pretrain
            self.R3D = R3D_model.R3DClassifier(num_classes=4, layer_sizes=(2, 2, 2, 2), pretrained=pretrain)
            self.modules = list(self.R3D.children())[:-1]
            linear_layer = nn.Linear(in_features=512, out_features=256)
            self.modules.append(linear_layer)
        elif classifier == "R2Plus1D":
            self.pretrain = pretrain
            self.R2Plus1D = R2Plus1D_model.R2Plus1DClassifier(num_classes=4, layer_sizes=(2, 2, 2, 2), pretrained=pretrain)
            self.modules = list(self.R2Plus1D.children())[:-1]
            linear_layer = nn.Linear(in_features=512, out_features=256)
            self.modules.append(linear_layer)
        elif classifier == "C3D":
            self.pretrain = pretrain
            self.C3D = C3D_model.C3D(num_classes=4, pretrained=pretrain)
            self.modules = list(self.C3D.children())[:-4]
            linear_layer = nn.Linear(in_features=256, out_features=256)
            self.modules.append(linear_layer)
        elif classifier == "PacR3D":
            self.pretrain = pretrain
            self.PacR3D = HM3D_model.R3DClassifier(num_classes=4, layer_sizes=(2, 2, 2, 2), pretrained=pretrain)
            self.modules = list(self.PacR3D.children())[:-1]
            linear_layer = nn.Linear(in_features=512, out_features=256)
            self.modules.append(linear_layer)
        self.model = nn.Sequential(*self.modules)
        self.relu = nn.ReLU()
        self.times = []
        self.dropout = nn.Dropout(0.5)

    def forward(self, images):
        print(self.model)
        features = self.model(images)
        return self.dropout(self.relu(features))
    
    def load_pretrained_weights(self, pretrained_weights_path):
        with tarfile.open(pretrained_weights_path, 'r') as tar:
            pretrained_weights = tar.extractfile('weights.pth')
        pretrained_weights = torch.load(pretrained_weights, map_location='cpu')
        self.R3D.load_state_dict(pretrained_weights)

class DecoderRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers=3):
        super(DecoderRNN, self).__init__()
        self.embed = nn.Embedding(vocab_size, embed_size)
        self.lstm = nn.LSTM(embed_size, hidden_size, num_layers)
        self.linear = nn.Linear(hidden_size, vocab_size)
        self.dropout = nn.Dropout(0.5)

    def forward(self, features, captions):
        embeddings = self.dropout(self.embed(captions))
        embeddings = torch.cat((features.unsqueeze(0), embeddings), dim=0)
        hiddens, _ = self.lstm(embeddings)
        outputs = self.linear(hiddens)
        return outputs


class CNNtoRNN(nn.Module):
    def __init__(self, embed_size, hidden_size, vocab_size, num_layers, classifier):
        super(CNNtoRNN, self).__init__()
        self.encoderCNN = EncoderCNN(classifier, embed_size)
        self.decoderRNN = DecoderRNN(embed_size, hidden_size, vocab_size, num_layers)

    def forward(self, video, captions):
        video = video.permute(0, 4, 1, 2, 3)
        features = self.encoderCNN(video)
        outputs = self.decoderRNN(features, captions)
        return outputs

    def caption_video(self, video, vocabulary, max_length=300):
        result_caption = []
        video = video.permute(0, 4, 1, 2, 3) 
        with torch.no_grad():
            x = self.encoderCNN(video).unsqueeze(0)
            states = None

            for _ in range(max_length):
                hiddens, states = self.decoderRNN.lstm(x, states)
                output = self.decoderRNN.linear(hiddens.squeeze(0))
                predicted = output.argmax(1)
                result_caption.append(predicted.item())
                x = self.decoderRNN.embed(predicted).unsqueeze(0)

                if vocabulary.itos[predicted.item()] == "<EOS>":
                    break

        return [vocabulary.itos[idx] for idx in result_caption]
