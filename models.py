import torch
import torch.nn as nn
import torch.nn.functional as F

# ==========================================
# 1. Artificial Neural Network (ANN)
# ==========================================
class ANN(nn.Module):
    def __init__(self, input_size, output_size=9, dropout=0.05):
        super(ANN, self).__init__()
        self.fc1 = nn.Linear(input_size, 18)
        self.fc2 = nn.Linear(18, 15)
        self.fc3 = nn.Linear(15, 12)
        self.fc4 = nn.Linear(12, output_size)
        
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        out = F.tanh(self.fc1(x))
        out = self.dropout(out)
        
        out = F.tanh(self.fc2(out))
        out = self.dropout(out)
        
        out = F.tanh(self.fc3(out))
                
        logits = self.fc4(out)
        return logits

# ==========================================
# 2. Bidirectional LSTM
# ==========================================
class BiLSTM(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BiLSTM, self).__init__()
        self.lstm = nn.LSTM(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        lstm_out, _ = self.lstm(x)
        out = F.relu(self.fc1(lstm_out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

# ==========================================
# 3. Bidirectional GRU
# ==========================================
class BiGRU(nn.Module):
    def __init__(self, input_size, hidden_size, output_size, num_layers=1, dropout=0.0):
        super(BiGRU, self).__init__()
        self.gru = nn.GRU(
            input_size=input_size, hidden_size=hidden_size, num_layers=num_layers, batch_first=True,
            bidirectional=True, dropout=dropout if num_layers > 1 else 0
        )
        self.dropout = nn.Dropout(dropout)
        self.fc1 = nn.Linear(hidden_size * 2, hidden_size)
        self.fc2 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        gru_out, _ = self.gru(x)
        out = F.relu(self.fc1(gru_out))
        out = self.dropout(out)
        logits = self.fc2(out)
        return logits

# ==========================================
# 4. 1D Res-ASPP U-Net
# ==========================================

# --- Helper: Transpose Wrapper ---
class ConvTransposeWrapper(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, stride=2, padding=0):
        super().__init__()
        self.deconv = nn.ConvTranspose1d(
            in_channels, out_channels, kernel_size, stride, padding=padding
        )

    def forward(self, x):
        x = x.transpose(1, 2)
        x = self.deconv(x)
        x = x.transpose(1, 2)
        return x

# --- Residual Block (Backbone) ---
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=1, stride=stride)
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.relu = nn.ReLU(inplace=True)
        
        self.conv2 = nn.Conv1d(out_channels, out_channels, kernel_size=3, padding=1)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.shortcut = nn.Sequential()
        if stride != 1 or in_channels != out_channels:
            self.shortcut = nn.Sequential(
                nn.Conv1d(in_channels, out_channels, kernel_size=1, stride=stride),
                nn.BatchNorm1d(out_channels)
            )

    def forward(self, x):
        x_in = x.transpose(1, 2)
        
        out = self.conv1(x_in)
        out = self.bn1(out)
        out = self.relu(out)
        
        out = self.conv2(out)
        out = self.bn2(out)
        
        residual = self.shortcut(x_in)
        
        out += residual
        out = self.relu(out)
        
        return out.transpose(1, 2)

# --- ASPP Block (Multi-Scale Context) ---
class ASPP(nn.Module):
    def __init__(self, in_channels, out_channels):
        super().__init__()
        self.conv1 = nn.Conv1d(in_channels, out_channels, kernel_size=1)
        self.bn1 = nn.BatchNorm1d(out_channels)
        
        self.conv2 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=6, dilation=6)
        self.bn2 = nn.BatchNorm1d(out_channels)
        
        self.conv3 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=12, dilation=12)
        self.bn3 = nn.BatchNorm1d(out_channels)
        
        self.conv4 = nn.Conv1d(in_channels, out_channels, kernel_size=3, padding=18, dilation=18)
        self.bn4 = nn.BatchNorm1d(out_channels)
        
        self.project = nn.Sequential(
            nn.Conv1d(out_channels * 4, out_channels, kernel_size=1),
            nn.BatchNorm1d(out_channels),
            nn.ReLU(inplace=True),
            nn.Dropout(0.5) 
        )

    def forward(self, x):
        x = x.transpose(1, 2) 
        
        y1 = F.relu(self.bn1(self.conv1(x)))
        y2 = F.relu(self.bn2(self.conv2(x)))
        y3 = F.relu(self.bn3(self.conv3(x)))
        y4 = F.relu(self.bn4(self.conv4(x)))
        
        res = torch.cat([y1, y2, y3, y4], dim=1)
        res = self.project(res)
        
        return res.transpose(1, 2)

# --- Main Res-U-Net + ASPP Architecture ---
class ResASPPUnet(nn.Module):
    def __init__(self, input_features=8, output_classes=9, dropout_rate=0.1):
        super().__init__()
        
        # --- Encoder ---
        self.enc1 = ResidualBlock(input_features, 16)
        self.pool1 = nn.MaxPool1d(kernel_size=2)
        
        self.enc2 = ResidualBlock(16, 32)
        self.pool2 = nn.MaxPool1d(kernel_size=2)
        
        self.enc3 = ResidualBlock(32, 64)
        self.pool3 = nn.MaxPool1d(kernel_size=2)
        
        # --- Bottleneck ---
        self.bottleneck = ASPP(in_channels=64, out_channels=128)
        
        # --- Decoder ---
        self.upconv3 = ConvTransposeWrapper(128, 64, kernel_size=2, stride=2)
        self.dec3 = ResidualBlock(128, 64) 
        self.drop3 = nn.Dropout(dropout_rate)
        
        self.upconv2 = ConvTransposeWrapper(64, 32, kernel_size=2, stride=2)
        self.dec2 = ResidualBlock(64, 32) 
        self.drop2 = nn.Dropout(dropout_rate)
        
        self.upconv1 = ConvTransposeWrapper(32, 16, kernel_size=2, stride=2)
        self.dec1 = ResidualBlock(32, 16) 
        self.drop1 = nn.Dropout(dropout_rate)
        
        # --- Output ---
        self.out_conv = nn.Conv1d(16, output_classes, kernel_size=1)
        
    def forward(self, x):
        # --- Encoder ---
        s1 = self.enc1(x)                
        p1 = self.pool1(s1.transpose(1, 2)).transpose(1, 2) 
        
        s2 = self.enc2(p1)               
        p2 = self.pool2(s2.transpose(1, 2)).transpose(1, 2) 
        
        s3 = self.enc3(p2)               
        p3 = self.pool3(s3.transpose(1, 2)).transpose(1, 2) 
        
        # --- Bottleneck ---
        b = self.bottleneck(p3)          
        
        # --- Decoder ---
        u3 = self.upconv3(b)
        diff3 = s3.size(1) - u3.size(1)
        if diff3 > 0: u3 = F.pad(u3, (0, 0, 0, diff3))
        
        c3 = torch.cat((u3, s3), dim=2)
        d3 = self.drop3(self.dec3(c3))
        
        u2 = self.upconv2(d3)
        diff2 = s2.size(1) - u2.size(1)
        if diff2 > 0: u2 = F.pad(u2, (0, 0, 0, diff2))
        
        c2 = torch.cat((u2, s2), dim=2)
        d2 = self.drop2(self.dec2(c2))
        
        u1 = self.upconv1(d2)
        diff1 = s1.size(1) - u1.size(1)
        if diff1 > 0: u1 = F.pad(u1, (0, 0, 0, diff1))
        
        c1 = torch.cat((u1, s1), dim=2)
        d1 = self.drop1(self.dec1(c1))
        
        # --- Output ---
        out = d1.transpose(1, 2)
        out = self.out_conv(out)
        out = out.transpose(1, 2)
        
        return out