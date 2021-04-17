import torch
from torch import nn
from torch.nn import functional as F
import math

from conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class L2L(nn.Module):
    def __init__(self):
        super(L2L,self).__init__()
        self.num_features=2+512
        self.audio_encoder = nn.Sequential(
            Conv2d(1, 32, kernel_size=3, stride=1, padding=1),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(32, 32, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(32, 64, kernel_size=3, stride=(3, 1), padding=1),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(64, 64, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(64, 128, kernel_size=3, stride=3, padding=1),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),
            Conv2d(128, 128, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(128, 256, kernel_size=3, stride=(3, 2), padding=1),
            Conv2d(256, 256, kernel_size=3, stride=1, padding=1, residual=True),

            Conv2d(256, 512, kernel_size=3, stride=1, padding=0),
            Conv2d(512, 512, kernel_size=1, stride=1, padding=0),)
        self.encoder_layer = nn.TransformerEncoderLayer(d_model=514, nhead=1)
        self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=6)
        #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.head = nn.Linear(self.num_features, 2)
    def forward(self, audio_sequences, landmark_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        # landmark_sequences =(B, T, 68, 4)
        B = audio_sequences.size(0)

        
        
        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)
        landmark_sequences = torch.cat([landmark_sequences[ :, i] for i in range(landmark_sequences.size(1))], dim=0)
        
        audio_embedding = self.audio_encoder(audio_sequences) # B, 512, 1, 1
        
        audio_embedding=audio_embedding.squeeze(3).transpose(1,2).repeat(1,136,1)#  B 68 512
        
        Input_tensor=torch.cat((audio_embedding,landmark_sequences),dim=2) #B 68 516
  
        x=self.transformer_encoder(Input_tensor)
        out=self.head(x)
        #print(x.shape)#80 68 2 
        out = torch.split(out, B, dim=0) # [(B, C, H, W)]
        
        outputs = torch.stack(out, dim=1)
        
        #print(outputs.shape)
        return outputs