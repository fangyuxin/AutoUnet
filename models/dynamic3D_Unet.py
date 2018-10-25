from AutoUnet import *

def EncoderBlock(ci, co): return nn.Sequential(CBR(ci, ci), CBR(ci, co), nn.MaxPool3d(2, 2))

class UnetEncoder(nn.Sequential):
    def __init__(self, len=3, ci=1, ch=32):
        layers = [btnk(ci, ch)]
        for _ in range(len):
            layers.append(EncoderBlock(ch, ch*2))
            ch *= 2

        super().__init__(*layers)



encoder = UnetEncoder()

m = AutoUnet(encoder, size=(32,16,48))

i = torch.zeros(1, 1, 32, 16, 48)
o = m(i)
print(o.shape)

for sz in m.dummy_size:
    print(sz)
    print('-'*100)