from AutoUnet import *


CBR = conv_BN_ReLU

def UnetBlock(ci, co): return nn.Sequential(CBR(ci, co), CBR(co, co), nn.MaxPool2d(2))

class UnetEncoder(nn.Sequential):
    def __init__(self, len=5, ci=1, ch=64):
        layers = [UnetBlock(ci, ch)[:-1]]
        for _ in range(len - 1):
            layers.append(UnetBlock(ch, ch*2))
            ch *= 2

        super().__init__(*layers)



m = UnetEncoder()

# for n in m:
#     print(n)
#     print('-'*100)

um = AutoUnet(m, 2)

#
# for n in um:
#     print(n)
#     print('-'*100)

for sz in um.dummy_size:
    print(sz)
    print('-'*100)

input = torch.zeros(1, 1, 256, 256)
output = um(input)

print(output.shape)

