import torch
import torch.nn as nn
import torch.nn.functional as F

class DeepAutoEncoder(nn.Module):
    def __init__(self, input_dim):
        super(DeepAutoEncoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 64),
            nn.ReLU(True),
            nn.Linear(64, 32)
        )
        self.decoder = nn.Sequential(
            nn.Linear(32, 64),
            nn.ReLU(True),
            nn.Linear(64, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim)
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class ConvAutoencoder(nn.Module):
    def __init__(self):
        super(ConvAutoencoder, self).__init__()
        # Encoder layers
        self.encoder = nn.Sequential(
            nn.Conv1d(in_channels=1, out_channels=16, kernel_size=3, stride=2, padding=1),  # Output: (16, 39)
            nn.ReLU(),
            nn.Conv1d(in_channels=16, out_channels=32, kernel_size=3, stride=2, padding=1),  # Output: (32, 20)
            nn.ReLU(),
            nn.Conv1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1),  # Output: (32, 10)
            nn.ReLU(),
            nn.Flatten(),
            nn.Linear(32 * 10, 32)
        )
        
        # Decoder layers
        self.decoder = nn.Sequential(
            nn.Linear(32, 32 * 10),
            nn.ReLU(),
            nn.Unflatten(1, (32, 10)),
            nn.ConvTranspose1d(in_channels=32, out_channels=32, kernel_size=3, stride=2, padding=1, output_padding=1),  # Output: (32, 20)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=32, out_channels=16, kernel_size=3, stride=2, padding=1),  # Output: (16, 39)
            nn.ReLU(),
            nn.ConvTranspose1d(in_channels=16, out_channels=1, kernel_size=2, stride=2, padding=0),  # Output: (1, 78)
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class VAE(nn.Module):
    def __init__(self, input_dim):
        super(VAE, self).__init__()
        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 128),
            nn.ReLU(True),
            nn.Linear(128, 128 // 2),
            nn.ReLU(True)
        )
        self.fc_mu = nn.Linear(128 // 2, 32)
        self.fc_logvar = nn.Linear(128 // 2, 32)
        self.decoder = nn.Sequential(
            nn.Linear(32, 128 // 2),
            nn.ReLU(True),
            nn.Linear(128 // 2, 128),
            nn.ReLU(True),
            nn.Linear(128, input_dim),
            nn.Sigmoid()
        )

    def encode(self, x):
        h = self.encoder(x)
        mu = self.fc_mu(h)
        logvar = self.fc_logvar(h)
        return mu, logvar

    def reparameterize(self, mu, logvar):
        std = torch.exp(0.5 * logvar)
        eps = torch.randn_like(std)
        return mu + eps * std

    def decode(self, z):
        return self.decoder(z)

    def forward(self, x):
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        return self.decode(z), mu, logvar

class Conv2DAutoencoder_v1(nn.Module):
    def __init__(self):
        super(Conv2DAutoencoder_v1, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(1, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 1, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(1),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x

class Conv2DAutoencoder(nn.Module):
    def __init__(self):
        super(Conv2DAutoencoder, self).__init__()
        self.encoder = nn.Sequential(
            nn.Conv2d(4, 32, 3, stride=2, padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.Conv2d(32, 64, 3, stride=2, padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.Conv2d(64, 128, 3, stride=2, padding=1),
            nn.BatchNorm2d(128),
            nn.ReLU()
        )
        self.decoder = nn.Sequential(
            nn.ConvTranspose2d(128, 64, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(64),
            nn.ReLU(),
            nn.ConvTranspose2d(64, 32, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(32),
            nn.ReLU(),
            nn.ConvTranspose2d(32, 4, 3, stride=2, padding=1, output_padding=1),
            nn.BatchNorm2d(4),
            nn.Sigmoid()
        )

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x
    
class SimpleMLP(nn.Module):
    def __init__(self, input_size, output_size):
        super(SimpleMLP, self).__init__()
        self.fc1 = nn.Linear(input_size, 512)
        self.fc2 = nn.Linear(512, 256)
        self.fc3 = nn.Linear(256, output_size)
        self.relu = nn.ReLU()
        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.dropout(x)
        x = self.relu(self.fc2(x))
        x = self.dropout(x)
        x = self.fc3(x)
        return x

class SimpleCNN(nn.Module):
    def __init__(self):
        super(SimpleCNN, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding='same')
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.flatten = nn.Flatten()
        self.fc1 = nn.Linear(64 * 5 * 2, 64) 
        self.fc2 = nn.Linear(64, 64)
        self.fc3 = nn.Linear(64, 32)

    def forward(self, x):
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        x = self.flatten(x)
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        return x

class SimpleLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleLSTM, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # LSTM Layer
        self.lstm = nn.LSTM(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 32)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.lstm(x, (h0, c0))
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class SimpleGRU(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleGRU, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # GRU layer
        self.gru = nn.GRU(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 32)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.gru(x, h0)
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out
    
class SimpleRNN(nn.Module):
    def __init__(self, input_size, hidden_size, num_layers):
        super(SimpleRNN, self).__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        
        # RNN Layer
        self.rnn = nn.RNN(input_size, hidden_size, num_layers, batch_first=True)
        
        # Fully connected layer
        self.fc = nn.Linear(hidden_size, 32)

    def forward(self, x):
        h0 = torch.zeros(self.num_layers, x.size(0), self.hidden_size).to(x.device)
        
        out, _ = self.rnn(x, h0)
        
        out = out[:, -1, :]
        out = self.fc(out)
        return out

class CNNLSTM(nn.Module):
    def __init__(self):
        super(CNNLSTM, self).__init__()
        # CNN layers
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=64, kernel_size=6, padding='same')
        self.bn1 = nn.BatchNorm1d(num_features=64)
        self.pool1 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv2 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding='same')
        self.bn2 = nn.BatchNorm1d(num_features=64)
        self.pool2 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=64, kernel_size=6, padding='same')
        self.bn3 = nn.BatchNorm1d(num_features=64)
        self.pool3 = nn.MaxPool1d(kernel_size=3, stride=2, padding=1)
        
        # Flatten layer to transition to the LSTM
        self.flatten = nn.Flatten()
        
        # LSTM layer
        # Assuming the output from CNN is a flattened vector, we reshape it back to sequence
        # LSTM input should be of shape (batch, seq_len, features)
        # We need to decide on seq_len and features
        # Here we use an example with seq_len = 5 and features = 64
        self.lstm = nn.LSTM(input_size=64, hidden_size=32, num_layers=1, batch_first=True)
        
        # Fully connected layers
        self.fc1 = nn.Linear(32, 64)  # Adjust the input size according to LSTM's hidden_size
        self.fc2 = nn.Linear(64, 32)

    def forward(self, x):
        # Pass data through CNN layers
        x = self.pool1(F.relu(self.bn1(self.conv1(x))))
        x = self.pool2(F.relu(self.bn2(self.conv2(x))))
        x = self.pool3(F.relu(self.bn3(self.conv3(x))))
        
        # Flatten and reshape for LSTM
        x = self.flatten(x)
        batch_size = x.shape[0]
        x = x.view(batch_size, 10, 64) 
        
        # Pass data through LSTM layers
        x, (hn, cn) = self.lstm(x)
        
        # We use the last hidden state to pass through the dense layers
        x = x[:, -1, :]
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x