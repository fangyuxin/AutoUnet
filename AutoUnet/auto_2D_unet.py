import torch
import torch.nn as nn
import numpy as np



def _get_sfs_idxs(szs):
    f_szs = [sz[-1] for sz in szs]
    sfs_idxs = list(np.where(np.array(f_szs[:-1]) != np.array(f_szs[1:]))[0]) + [len(f_szs) - 1]
    return sfs_idxs

def cut_model(m, cut):
    return nn.Sequential(*(list(m.children())[:cut]))

flatten_model=lambda m: sum(map(flatten_model, m.children()),[]) if len(list(m.children())) else [m]

def first_ch(m):
    for l in flatten_model(m):
        if hasattr(l, 'weight'): return l.weight.size(1)

def mk_res_encoder(res_m):
    return cut_model(res_m, -2)

def mk_dse_encoder(dse_m):
    return nn.Sequential(nn.Conv2d(3, 64, 1), *list(dse_m.children())[0][4:-1])


class Hook():

    def __init__(self, m:nn.Module, hook_func, is_forward=True):
        self.hook_func = hook_func
        self.stored = None
        f = m.register_forward_hook if is_forward else m.register_backward_hook
        self.hook = f(self.hook_fn)
        self.removed = False

    def hook_fn(self, m:nn.Module, input, output):
        input = (o.detach() for o in input) if isinstance(input, (tuple, list)) else input.detach()
        output = (o.detach() for o in output) if isinstance(output, (tuple, list)) else output.detach()
        self.stored = self.hook_func(m, input, output)

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True


class Hooks():
    def __init__(self, ms, hook_func, is_forward=True):
        self.hooks = [Hook(m, hook_func, is_forward) for m in ms]

    def __getitem__(self, i): return self.hooks[i]
    def __len__(self): return len(self.hooks)
    def __iter__(self):return iter(self.hooks)

    @property
    def stored(self): return [o.stored for o in self]

    def remove(self):
        for h in self: h.remove()

def hook_output(m): return Hook(m, lambda m, i, o: o)
def hook_outputs(ms): return Hooks(ms, lambda m, i, o: o)


def model_sizes(m:nn.Module, size=(224, 224)):
    hooks = hook_outputs(m.children())
    ch_in = first_ch(m)
    dummy = torch.zeros(1, ch_in, *size)
    dummy = m.eval()(dummy)
    res = [o.stored.shape for o in hooks]
    return res, dummy, hooks



def up_conv(ch_in, ch_out, hook=None, f_size_in=None, ks=2, s=2, p=0, op=0, bias=False):
    if hook is not None and f_size_in is not None:
        f_size_out = hook.stored.size(-1)
        s = f_size_out//(f_size_in - 1)
        ks = s
        op = f_size_out - (f_size_in - 1)*s + 2*p - ks
    else:
        pass
    return nn.ConvTranspose2d(ch_in, ch_out, ks, s, p, op, bias=bias)



def conv_BN_ReLU(ch_in, ch_out, ks=3, s=1, p=1, bn=True, bias=False):
    layers = [nn.Conv2d(ch_in, ch_out, ks, s, p, bias=bias)]
    if bn: layers.append(nn.BatchNorm2d(ch_out))
    layers.append(nn.ReLU(True))
    return nn.Sequential(*layers)


class AutoUnetBlock(nn.Module):

    def __init__(self, up_ch, skip_ch, hook, up_f_sz):
        super().__init__()
        self.hook = hook
        ch_in = up_ch
        self.up_conv = up_conv(ch_in, skip_ch, hook, up_f_sz)
        ch_in = skip_ch*2
        self.CBR_1 = conv_BN_ReLU(ch_in, ch_in//2)
        ch_in = ch_in//2
        self.CBR_2 = conv_BN_ReLU(ch_in, ch_in)

    def forward(self, up_in):
        up_out = self.up_conv(up_in)
        cat_x = torch.cat([self.hook.stored, up_out], dim=1)
        return self.CBR_2(self.CBR_1(cat_x))



class AutoUnet(nn.Sequential):

    def __init__(self, encoder, n_classes, imsize=(256, 256)):
        sfs_szs, dummy, self.sfs = model_sizes(encoder, imsize)
        sfs_idxs = _get_sfs_idxs(sfs_szs)
        sfs_idxs.reverse()

        ch_in = sfs_szs[-1][1]
        center = nn.Sequential(conv_BN_ReLU(ch_in, ch_in*2), conv_BN_ReLU(ch_in*2, ch_in*2))
        dummy = center(dummy)
        layers = [encoder, nn.ReLU(True), center]
        dummy_szs = ['***center***', dummy.size(), '***center***']



        for i in sfs_idxs:
            up_ch, up_f_sz, skip_ch = dummy.size(1), dummy.size(-1), sfs_szs[i][1]
            unet_block = AutoUnetBlock(up_ch, skip_ch, self.sfs[i], up_f_sz)
            layers.append(unet_block)
            dummy = unet_block(dummy)

            dummy_szs.append(dummy.size())
            dummy_szs.insert(0, sfs_szs[i])

        n = unet_block.CBR_2[0].out_channels
        layers.append(nn.Conv2d(n, n_classes, 1))
        super().__init__(*layers)


        self.dummy_size = dummy_szs


