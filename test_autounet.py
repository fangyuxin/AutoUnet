
from auto_unet import *
from torchvision.models import *



res_m = ResNet(resnet.BasicBlock, [2, 4, 6, 11])
dse_m = DenseNet(num_init_features=64, growth_rate=32, block_config=(2, 4, 16, 10, 5))

res_encoder = mk_res_encoder(res_m)
dse_encoder = mk_dse_encoder(dse_m)


# print(first_ch(resnet))
# print(first_ch(densenet))


# for i, m in enumerate(flatten_model(resnet)):
#     print('{}: {}'.format(i, m))
#     print('-'*100)

# for i, m in enumerate(flatten_model(densenet)):
#     print('{}: {}'.format(i, m))
#     print('-'*100)



ResUnet = AutoUnet(res_encoder, 2)
DenseUnet = AutoUnet(dse_encoder, 2)


for sz in ResUnet.dummy_size:
    print(sz)

print('-'*100)

for sz in DenseUnet.dummy_size:
    print(sz)