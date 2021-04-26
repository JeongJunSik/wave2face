import torch
from torch import nn
from torch.nn import functional as F
import math
from timm.models.layers import DropPath, to_2tuple, trunc_normal_
from conv import Conv2dTranspose, Conv2d, nonorm_Conv2d

class L2L(nn.Module):
    def __init__(self):
        super(L2L,self).__init__()
        self.num_features=4
        self.pos_embed = nn.Parameter(torch.zeros(1, 68, 4))

        self.pos_drop = nn.Dropout(p=0.1)
        trunc_normal_(self.pos_embed, std=.02)
        
        
        
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
        self.encoder_layer1 = nn.TransformerEncoderLayer(d_model=68, nhead=2)
        #self.encoder_layer2 = nn.TransformerEncoderLayer(d_model=208, nhead=2)
        self.transformer_encoder1 = nn.TransformerEncoder(self.encoder_layer1, num_layers=3)
        #self.transformer_encoder2 = nn.TransformerEncoder(self.encoder_layer2, num_layers=3)
        #self.transformer_encoder = nn.TransformerEncoder(self.encoder_layer, num_layers=3)
        self.head1 = nn.Linear(self.num_features, 2)
        self.mlp=nn.Linear(512,40)# 입부분만
        #self.head2 = nn.Linear(208, 2)
    def forward(self, audio_sequences, landmark_sequences):
        # audio_sequences = (B, T, 1, 80, 16)
        # landmark_sequences =(B, T, 68, 4)
        
        #0~17  48 68 
        b=audio_sequences.size(0)
        
        audio_sequences = torch.cat([audio_sequences[:, i] for i in range(audio_sequences.size(1))], dim=0)# 512 1 1 
        landmark_sequences = torch.cat([landmark_sequences[ :, i] for i in range(landmark_sequences.size(1))], dim=0)# 80 136 2 
        B = landmark_sequences.shape[0]
        audio_embedding = self.audio_encoder(audio_sequences).squeeze().unsqueeze(1) # B, 1, 512, 
        audio=self.mlp(audio_embedding).view(B,-1,2)
       
        landmark_sequences[:,48:68,0:2]=landmark_sequences[:,48:68,0:2]+audio #B 68 516
        Input_tensor =landmark_sequences+  self.pos_embed
        #Input_tensor[:,0:68,:] = Input_tensor[:,0:68,:].detach().clone() + self.pos_embed
        #Input_tensor[:,68:,:] = Input_tensor[:,68:,:].detach().clone() + self.pos_embed
        #Input_tensor=self.pos_drop(Input_tensor)
        x=self.transformer_encoder1(Input_tensor.transpose(1,2)).transpose(1,2) # B,68,4
        out=self.head1(x)
        
        #x2=self.transformer_encoder2(out)
        #out2=self.head2(x2)
        #print(x.shape)#80 68 2 
        out = torch.split(out, b, dim=0) # [(B, C, H, W)]
        
        outputs = torch.stack(out, dim=1)
        #import pdb;pdb.set_trace()
        #print(outputs.shape)
        return outputs
