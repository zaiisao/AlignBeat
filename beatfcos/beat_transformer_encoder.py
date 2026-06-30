import torch
from torch import nn
import sys
from beatfcos.DilatedTransformerLayer import DilatedTransformerLayer

class BeatTransformerEncoder(nn.Module):
    """
    Input : (B, T, 128) mel spectrogram
    Output : C1(B, T, d), C2(B, T, d), C3(B, T, d)
    tap at DSA layer 2, 5, 8 (3 groups of 3)
    """
    def __init__(self, attn_len=5, dmodel=128, nhead=2, d_hid=512,
                nlayers=9, norm_first=True, dropout=0.1):
        
        super().__init__()
        assert dmodel % nhead == 0

        self.conv1 = nn.Conv2d(1, 32, kernel_size=(5,3), stride=1, padding=(2,0))
        self.maxpool1 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.dropout1 = nn.Dropout(p=dropout)

        self.conv2 = nn.Conv2d(32, 64, kernel_size=(1, 12), stride=1, padding=(0,0))
        self.maxpool2 = nn.MaxPool2d(kernel_size=(1,3), stride=(1,3))
        self.dropout2 = nn.Dropout(p=dropout)

        self.conv3 = nn.Conv2d(64, dmodel, kernel_size=(3, 6), stride=1, padding=(1,0))
        self.maxpool3 = nn.MaxPool2d(kernel_size=(1, 3), stride=(1, 3))
        self.dropout3 = nn.Dropout(p=dropout)

        self.nlayers = nlayers
        self.dsa_layers = nn.ModuleList([
            DilatedTransformerLayer(
                dmodel, nhead, d_hid, dropout,
                Er_provided=False, attn_len=attn_len, norm_first=norm_first
            )
            for _ in range(nlayers)
        ])

    def forward(self, x):
            x = x.unsqueeze(1)

            x = torch.relu(self.maxpool1(self.conv1(x)))
            x = self.dropout1(x)
            x = torch.relu(self.maxpool2(self.conv2(x)))
            x = self.dropout2(x)
            x = torch.relu(self.maxpool3(self.conv3(x)))
            x = self.dropout3(x)

            x = x.transpose(1, 3).squeeze(1).contiguous()  # (B, T, dmodel)

            for i, layer in enumerate(self.dsa_layers):
                x, _ = layer(x, layer=i)
                if i == 2: C1 = x
                if i == 5: C2 = x
                if i == 8: C3 = x

            return C1, C2, C3