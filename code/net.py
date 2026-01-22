class unet_block(nn.Module):
    def __init__(self, in_ch, out_ch, down=True):
        super(unet_block, self).__init__()
        self.pool = nn.MaxPool2d(2) if down else nn.Identity()
        self.block = nn.Sequential(
            nn.Conv2d(in_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True),
            nn.Conv2d(out_ch, out_ch, 3, padding=1),
            nn.BatchNorm2d(out_ch),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        x = self.pool(x)
        return self.block(x)

class DenseLayer(nn.Module):
    def __init__(self, in_channels, growth_rate):
        super(DenseLayer, self).__init__()
        self.layer = nn.Sequential(
            nn.Conv2d(in_channels, growth_rate, kernel_size=3, padding=1, bias=False),
            nn.BatchNorm2d(growth_rate),
            nn.ReLU(inplace=True)
        )
    def forward(self, x):
        return torch.cat([x, self.layer(x)], 1)

class dense_unet_up(nn.Module):
    def __init__(self, in_ch, out_ch, growth_rate=32, num_layers=4):
        super(dense_unet_up, self).__init__()
        self.up = nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True)
        self.dense_block = nn.Sequential(*[
            DenseLayer(in_ch + i * growth_rate, growth_rate) for i in range(num_layers)
        ])
        self.conv = nn.Conv2d(in_ch + growth_rate * num_layers, out_ch, kernel_size=1)

    def forward(self, x1, x2):
        x1 = self.up(x1)
        # Pad if needed
        diffY = x2.size()[2] - x1.size()[2]
        diffX = x2.size()[3] - x1.size()[3]
        x1 = F.pad(x1, (diffX // 2, diffX - diffX // 2, diffY // 2, diffY - diffY // 2))
        x = torch.cat([x2, x1], dim=1)
        x = self.dense_block(x)
        x = self.conv(x)
        return x

class DenseUnet(nn.Module):
    def __init__(self, growth_rate=32, num_layers=4):
        super(DenseUnet, self).__init__()
        # Encoder
        self.inc = unet_block(1, 64, down=False)
        self.down1 = unet_block(64, 128)
        self.down2 = unet_block(128, 256)
        self.down3 = unet_block(256, 512)
        self.down4 = unet_block(512, 512)

        self.xmamba_s1 = XMambaLayer(dim=64)
        self.xmamba_s2 = XMambaLayer(dim=128)
        self.xmamba_s3 = XMambaLayer(dim=256)
        self.xmamba_s4 = XMambaLayer(dim=512)
        
        # SMamba for Bottleneck
        self.smamba_bottleneck = SMambaLayer(dim=512)
        # ----------------------------------------------

        # Decoder
        self.up1 = dense_unet_up(1024, 256, growth_rate, num_layers)
        self.up2 = dense_unet_up(512, 128, growth_rate, num_layers)
        self.up3 = dense_unet_up(256, 64, growth_rate, num_layers)
        self.up4 = dense_unet_up(128, 64, growth_rate, num_layers)
        self.outc = nn.Conv2d(64, 1, 1)
    
    def forward(self, x):
        # Encoder Path
        x1 = self.inc(x)         # [B, 64, H, W]
        x1_m = self.xmamba_s1(x1)
        
        x2 = self.down1(x1)      # [B, 128, H/2, W/2]
        x2_m = self.xmamba_s2(x2)
        
        x3 = self.down2(x2)      # [B, 256, H/4, W/4]
        x3_m = self.xmamba_s3(x3)
        
        x4 = self.down3(x3)      # [B, 512, H/8, H/8]
        x4_m = self.xmamba_s4(x4)
        
        x5 = self.down4(x4)      # [B, 512, H/16, W/16]
        
        # Bottleneck (Smamba)
        x5_m = self.smamba_bottleneck(x5)
        
        # Decoder Path (Using Mamba-enhanced features for skip connection)
        x = self.up1(x5_m, x4_m) 
        x = self.up2(x, x3_m)
        x = self.up3(x, x2_m)
        x = self.up4(x, x1_m)
        
        x = self.outc(x)
        return x