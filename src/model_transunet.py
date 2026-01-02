import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange, repeat
from einops.layers.torch import Rearrange

# === TransUNet 核心组件 ===

class DoubleConv(nn.Module):
    """(convolution => [BN] => ReLU) * 2"""
    def __init__(self, in_channels, out_channels, mid_channels=None):
        super().__init__()
        if not mid_channels:
            mid_channels = out_channels
        self.double_conv = nn.Sequential(
            nn.Conv2d(in_channels, mid_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(mid_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(mid_channels, out_channels, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True)
        )

    def forward(self, x):
        return self.double_conv(x)

class PatchEmbedding(nn.Module):
    def __init__(self, image_size, patch_size, in_channels, embed_dim):
        super().__init__()
        self.num_patches = (image_size // patch_size) ** 2
        self.proj = nn.Sequential(
            nn.Conv2d(in_channels, embed_dim, kernel_size=patch_size, stride=patch_size),
            Rearrange('b c h w -> b (h w) c'),
        )
        self.pos_embedding = nn.Parameter(torch.randn(1, self.num_patches + 1, embed_dim))
        self.cls_token = nn.Parameter(torch.randn(1, 1, embed_dim))

    def forward(self, x):
        b = x.shape[0]
        x = self.proj(x)
        cls_tokens = repeat(self.cls_token, '() n d -> b n d', b=b)
        x = torch.cat([cls_tokens, x], dim=1)
        x += self.pos_embedding[:, :x.size(1)]
        return x

class MultiHeadAttention(nn.Module):
    def __init__(self, dim, num_heads, dropout=0.):
        super().__init__()
        self.num_heads = num_heads
        self.scale = (dim // num_heads) ** -0.5
        self.qkv = nn.Linear(dim, dim * 3, bias=False)
        self.proj = nn.Linear(dim, dim)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        b, n, c = x.shape
        qkv = self.qkv(x).reshape(b, n, 3, self.num_heads, c // self.num_heads).permute(2, 0, 3, 1, 4)
        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.dropout(attn)
        x = (attn @ v).transpose(1, 2).reshape(b, n, c)
        return self.proj(x)

class TransformerEncoder(nn.Module):
    def __init__(self, dim, num_heads, mlp_dim, dropout=0.):
        super().__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.attn = MultiHeadAttention(dim, num_heads, dropout)
        self.norm2 = nn.LayerNorm(dim)
        self.mlp = nn.Sequential(
            nn.Linear(dim, mlp_dim),
            nn.GELU(),
            nn.Dropout(dropout),
            nn.Linear(mlp_dim, dim),
            nn.Dropout(dropout)
        )

    def forward(self, x):
        x = x + self.attn(self.norm1(x))
        x = x + self.mlp(self.norm2(x))
        return x

class Up(nn.Module):
    def __init__(self, in_channels, out_channels, bilinear=True):
        super().__init__()
        if bilinear:
            self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
            self.conv = DoubleConv(in_channels, out_channels, in_channels // 2)
        else:
            self.up = nn.ConvTranspose2d(in_channels, in_channels // 2, kernel_size=2, stride=2)
            self.conv = DoubleConv(in_channels, out_channels)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = torch.nn.functional.pad(x1, [diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2])
        x = torch.cat([x2, x1], dim=1)
        return self.conv(x)

class TransUNet(nn.Module):
    def __init__(self, img_size=512, in_channels=3, out_channels=1, 
                 embed_dim=512, num_heads=8, mlp_dim=2048, num_layers=4, patch_size=16):
        super().__init__()
        self.img_size = img_size
        self.patch_size = patch_size
        
        # Encoder (CNN)
        filters = [64, 128, 256, 512]
        self.inc = DoubleConv(in_channels, filters[0])
        self.down1 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(filters[0], filters[1]))
        self.down2 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(filters[1], filters[2]))
        self.down3 = nn.Sequential(nn.MaxPool2d(2), DoubleConv(filters[2], filters[3]))
        
        # Transformer Bridge
        self.patch_embed = PatchEmbedding(img_size//8, patch_size//8, filters[3], embed_dim)
        self.transformer_encoders = nn.ModuleList([
            TransformerEncoder(embed_dim, num_heads, mlp_dim) for _ in range(num_layers)
        ])
        self.transformer_conv = nn.Conv2d(embed_dim, filters[3], kernel_size=3, padding=1)
        
        # Decoder
        self.up1 = Up(filters[3] + filters[2], filters[2])
        self.up2 = Up(filters[2] + filters[1], filters[1])
        self.up3 = Up(filters[1] + filters[0], filters[0])
        self.outc = nn.Conv2d(filters[0], out_channels, kernel_size=1)

    def forward(self, x):
        # CNN Encoder
        x1 = self.inc(x)        # 64, H, W
        x2 = self.down1(x1)     # 128, H/2, W/2
        x3 = self.down2(x2)     # 256, H/4, W/4
        x4 = self.down3(x3)     # 512, H/8, W/8
        
        # Transformer
        emb = self.patch_embed(x4)
        for enc in self.transformer_encoders:
            emb = enc(emb)
            
        tok = emb[:, 1:, :] # Remove CLS token -> [B, 1024, 512]
        
        # === 修复 Shape Mismatch 的关键逻辑 ===
        # 1. 计算实际的网格大小 (sqrt(1024) = 32)
        n_tokens = tok.shape[1]
        grid_size = int(n_tokens ** 0.5)
        
        # 2. 还原回空间特征图 [B, 512, 32, 32]
        feat = rearrange(tok, 'b (h w) c -> b c h w', h=grid_size, w=grid_size)
        
        # 3. 插值上采样回 x4 的尺寸 [B, 512, 64, 64]
        # 这样才能和后面的卷积层以及 Decoder 兼容
        feat = F.interpolate(feat, size=(x4.shape[2], x4.shape[3]), mode='bilinear', align_corners=False)
        
        # 4. 卷积投影
        feat = self.transformer_conv(feat)
        
        # Decoder
        x = self.up1(feat, x3)
        x = self.up2(x, x2)
        x = self.up3(x, x1)
        logits = self.outc(x)
        
        return logits