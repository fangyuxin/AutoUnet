import torch
import torch.nn as nn

def btnk(ch_i, ch_o): return nn.Conv3d(ch_i, ch_o, 1)

def CBR(ch_i, ch_o, ks=3, s=1, p=1):
    return nn.Sequential(nn.Conv3d(ch_i, ch_o, ks, s, p), nn.BatchNorm3d(ch_o), nn.ReLU())

def down(): return nn.MaxPool3d(2, 2)

def up(ch): return nn.ConvTranspose3d(ch, ch, 2, 2)

def block(ch, phase):
    ch1, ch2, ch3 = (ch, ch, ch*2) if phase == 'down' else (int(ch*(3/2)), int(ch/2), int(ch/2))
    return nn.Sequential(CBR(ch1, ch2), CBR(ch2, ch3))

class Unet(nn.Module):
    def __init__(self, ch=32, len=3, n_cls=2):
        super().__init__()

        self.len = len
        self.o = []

        bt_lrs = [btnk(1, ch)]
        spl_lrs = []
        block_lrs = []

        for phase in ['down', 'up']:
            for _ in range(len):
                block_lrs.append(block(ch, phase))
                spl_lrs.append(down() if phase == 'down' else up(ch))
                ch = ch*2 if phase == 'down' else int(ch/2)
            if phase == 'down':
                block_lrs.append(block(ch, phase))
                spl_lrs.append('_')
                ch = ch*2
        bt_lrs.append(btnk(ch, n_cls))

        self.block_lrs = block_lrs
        self.spl_lrs = spl_lrs
        self.bt_lrs = bt_lrs



    def forward(self, x):
        x = self.bt_lrs[0](x)
        for l in range(self.len):
            x = self.block_lrs[l](x)
            self.o.append(x)
            x = self.spl_lrs[l](x)
        x = self.block_lrs[self.len](x)
        for l in range(self.len + 1, 2*self.len + 1):
            x = self.spl_lrs[l](x)
            x = torch.cat((self.o[2*self.len - l], x), dim=1)
            x = self.block_lrs[l](x)
        x = self.bt_lrs[1](x)

        return x


m = Unet()
i = torch.zeros(1, 1, 32, 16, 48)
o = m(i)
print(o.shape)









