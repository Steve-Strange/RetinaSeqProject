import torch
import torch.nn as nn
import torch.nn.functional as F

# ... (在此处粘贴 FocalModulation, FocalModulationContextAggregation, VisionMambaInspired 类) ...
# 为了节省篇幅，这里假设上述类已经包含，仅展示核心整合部分

class FocalModulation(nn.Module):
    def __init__(self, in_channels, gamma=2.0, alpha=0.25):
        super(FocalModulation, self).__init__()
        self.gamma = gamma
        self.alpha = alpha
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)
        self.conv = nn.Conv2d(in_channels, in_channels, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        mean = self.gap(x)
        max_val = self.gmp(x)
        modulation = (max_val - mean) * self.alpha
        modulation = self.conv(modulation)
        modulation = self.sigmoid(modulation)
        scaled_inputs = x * modulation
        outputs = torch.pow(scaled_inputs, self.gamma)
        return outputs

class FocalModulationContextAggregation(nn.Module):
    def __init__(self, in_channels, filters):
        super(FocalModulationContextAggregation, self).__init__()
        self.conv1 = nn.Conv2d(in_channels, filters, kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.relu2 = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.conv_ctx = nn.Conv2d(filters, filters, kernel_size=1)
        self.sigmoid = nn.Sigmoid()
        self.focal_mod = FocalModulation(filters)

    def forward(self, x):
        c1 = self.relu1(self.conv1(x))
        c2 = self.relu2(self.conv2(x))
        global_context = self.gap(c2)
        global_context = self.sigmoid(self.conv_ctx(global_context))
        global_context = c1 * global_context
        fm = self.focal_mod(global_context)
        return torch.cat([c1, fm], dim=1)

class VisionMambaInspired(nn.Module):
    def __init__(self, dim, dropout_rate=0.1):
        super(VisionMambaInspired, self).__init__()
        self.norm1 = nn.LayerNorm(dim)
        self.token_mixer = nn.Sequential(
            nn.Conv2d(dim, dim, kernel_size=3, padding=1, groups=dim),
            nn.GELU(),
            nn.Dropout(dropout_rate)
        )
        self.norm2 = nn.LayerNorm(dim)
        self.channel_mixer = nn.Sequential(
            nn.Linear(dim, dim * 4),
            nn.GELU(),
            nn.Dropout(dropout_rate),
            nn.Linear(dim * 4, dim),
            nn.Dropout(dropout_rate)
        )

    def forward(self, x):
        b, c, h, w = x.shape
        shortcut = x
        x_perm = x.permute(0, 2, 3, 1)
        x_norm = self.norm1(x_perm).permute(0, 3, 1, 2)
        x_tm = self.token_mixer(x_norm) + shortcut
        shortcut = x_tm
        x_perm = x_tm.permute(0, 2, 3, 1)
        x_norm = self.norm2(x_perm)
        x_cm = self.channel_mixer(x_norm)
        x_cm = x_cm.permute(0, 3, 1, 2)
        return x_cm + shortcut

class LiteFusionAttention(nn.Module):
    def __init__(self, in_channels, filters):
        super(LiteFusionAttention, self).__init__()
        self.proj1 = nn.Conv2d(in_channels, filters, kernel_size=1)
        self.norm = nn.LayerNorm(filters)
        self.conv = nn.Conv2d(filters, filters, kernel_size=3, padding=1)
        self.fmca = FocalModulationContextAggregation(filters, filters)
        self.proj2 = nn.Conv2d(2 * filters, filters, kernel_size=1)
        self.vm = VisionMambaInspired(filters)
        self.res_proj = nn.Conv2d(in_channels, filters, kernel_size=1) if in_channels != filters else nn.Identity()

    def forward(self, x):
        input_tensor = self.proj1(x)
        x_perm = input_tensor.permute(0, 2, 3, 1)
        x_norm = self.norm(x_perm).permute(0, 3, 1, 2)
        x_conv = self.conv(x_norm)
        x_fmca = self.fmca(x_conv)
        x_proj = self.proj2(x_fmca)
        res = self.res_proj(x) if isinstance(self.res_proj, nn.Conv2d) else input_tensor
        out = x_proj + res
        out = self.vm(out)
        return out

class RA_AttentionBlock(nn.Module):
    def __init__(self, in_channels, n_classes, k):
        super(RA_AttentionBlock, self).__init__()
        self.k = k
        self.n_classes = n_classes
        self.conv = nn.Conv2d(in_channels, k * n_classes, kernel_size=3, padding=1)
        self.bn = nn.BatchNorm2d(k * n_classes)
        self.relu = nn.ReLU()
        self.gap = nn.AdaptiveAvgPool2d(1)
        self.gmp = nn.AdaptiveMaxPool2d(1)

    def forward(self, x):
        b, c, h, w = x.shape
        f = self.relu(self.bn(self.conv(x)))
        x1 = self.gmp(f)
        x2 = self.gap(f)
        x_mul = x1 * x2
        x_reshape = x_mul.view(b, self.n_classes, self.k)
        s = torch.mean(x_reshape, dim=-1, keepdim=False)
        f_perm = f.permute(0, 2, 3, 1)
        f_reshape = f_perm.view(b, h, w, self.n_classes, self.k)
        f_mean = torch.mean(f_reshape, dim=-1, keepdim=False)
        s_expanded = s.view(b, 1, 1, self.n_classes)
        x_weighted = f_mean * s_expanded
        m = torch.mean(x_weighted, dim=-1, keepdim=True)
        m = m.permute(0, 3, 1, 2)
        semantic = x * m
        return semantic

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, dropout_rate=0.5):
        super(ConvBlock, self).__init__()
        self.conv1x1 = nn.Conv2d(in_channels, out_channels, kernel_size=1, padding=0, bias=False)
        self.conv3x3 = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=1, bias=False)
        self.conv3x3_dilated = nn.Conv2d(in_channels, out_channels, kernel_size=3, padding=2, dilation=2, bias=False)
        self.leaky_relu = nn.LeakyReLU(0.1)

    def forward(self, x):
        c1 = self.conv1x1(x)
        c3 = self.conv3x3(x)
        c3d = self.conv3x3_dilated(x)
        out = c1 + c3 + c3d
        out = self.leaky_relu(out)
        return out

class LFANet(nn.Module):
    def __init__(self, input_channels=3, num_classes=1, feature_scale=2, dropout=0.5):
        super(LFANet, self).__init__()
        filters = [int(x / feature_scale) for x in [16, 32, 64]]
        
        self.conv1 = ConvBlock(input_channels, filters[0], dropout)
        self.pool1 = nn.MaxPool2d(2)
        self.bn1 = nn.BatchNorm2d(filters[0])
        
        self.conv2 = ConvBlock(filters[0], filters[1], dropout)
        self.pool2 = nn.MaxPool2d(2)
        self.bn2 = nn.BatchNorm2d(filters[1])
        
        self.conv3 = ConvBlock(filters[1], filters[2], dropout)
        self.pool3 = nn.MaxPool2d(2)
        self.bn3 = nn.BatchNorm2d(filters[2])
        
        self.lfa = LiteFusionAttention(filters[2], filters=32)
        
        lfa_out_channels = 32 
        self.att1 = RA_AttentionBlock(lfa_out_channels, 1, 16)
        
        self.up1 = nn.ConvTranspose2d(lfa_out_channels * 2, filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.att2 = RA_AttentionBlock(filters[1], 1, 16)
        self.dec_conv1 = nn.Conv2d(filters[1] + filters[2], filters[2], kernel_size=3, padding=1)
        self.relu1 = nn.ReLU()
        
        self.up2 = nn.ConvTranspose2d(filters[2], filters[2], kernel_size=3, stride=2, padding=1, output_padding=1)
        
        self.att3 = RA_AttentionBlock(filters[0], 1, 16)
        self.dec_conv2 = nn.Conv2d(filters[0] + filters[2], filters[2], kernel_size=3, padding=1)
        self.relu2 = nn.ReLU()
        
        self.up3 = nn.ConvTranspose2d(filters[2], filters[0], kernel_size=3, stride=2, padding=1, output_padding=1)
        self.dec_conv3 = nn.Conv2d(filters[0], filters[0], kernel_size=3, padding=1)
        self.relu3 = nn.ReLU()
        
        self.final = nn.Conv2d(filters[0], num_classes, kernel_size=1)

    def forward(self, x):
        c1 = self.conv1(x)
        p1 = self.bn1(self.pool1(c1))
        c2 = self.conv2(p1)
        p2 = self.bn2(self.pool2(c2))
        c3 = self.conv3(p2)
        p3 = self.bn3(self.pool3(c3))
        lfa = self.lfa(p3)
        att1 = self.att1(lfa)
        fused = torch.cat([att1, lfa], dim=1)
        d1 = self.up1(fused)
        if d1.size() != c2.size(): d1 = F.interpolate(d1, size=c2.shape[2:], mode='bilinear', align_corners=True)
        att2 = self.att2(c2)
        d1 = torch.cat([att2, d1], dim=1)
        d1 = self.relu1(self.dec_conv1(d1))
        d2 = self.up2(d1)
        if d2.size() != c1.size(): d2 = F.interpolate(d2, size=c1.shape[2:], mode='bilinear', align_corners=True)
        att3 = self.att3(c1)
        d2 = torch.cat([att3, d2], dim=1)
        d2 = self.relu2(self.dec_conv2(d2))
        d3 = self.up3(d2)
        if d3.size() != x.size(): d3 = F.interpolate(d3, size=x.shape[2:], mode='bilinear', align_corners=True)
        d3 = self.relu3(self.dec_conv3(d3))
        out = self.final(d3)
        return out