import torch
import torch.nn as nn
import numpy as np


flatten_model=lambda m: sum(map(flatten_model, m.children()),[]) if len(list(m.children())) else [m]

def first_ch_sz(model):
    for l in flatten_model(model):
        if hasattr(l, 'weight'): return l.weight.size(1)

def _szdiff_idxs(szs):
    f_szs = [sz[-1] for sz in szs]
    return list(np.where(np.array(f_szs[:-1]) != np.array(f_szs[1:]))[0]) + [len(f_szs) - 1]

class Hook():
    def __init__(self, m, hook_func, is_forward=True):
        phase = m.register_forward_hook if is_forward else m.register_backward_hook
        self.output = None
        self.hook_func = hook_func
        self.hook = phase(self._hook_func)
        self.removed = False

    def _hook_func(self, m, input, output):
        input = (i.detach() for i in input) if isinstance(input, (list, tuple)) else input.detach()
        output = (o.detach() for o in output) if isinstance(output, (list, tuple)) else output.detach()
        self.output = self.hook_func(m, input, output)

    def remove(self):
        if not self.removed:
            self.hook.remove()
            self.removed = True

class Hooks():
    def __init__(self, model, hook_func, is_forward=True):
        self.hooks = [Hook(m, hook_func, is_forward) for m in model.children()]

    def __getitem__(self, i): return self.hooks[i]
    def __len__(self): return len(self.hooks)
    def __iter__(self): return iter(self.hooks)

    @property
    def outputs(self): return [hook.output for hook in self]

    def removed(self):
        for hook in self: hook.remove()

def get_hook_output(m): return Hook(m, lambda m, input, output: output)

def get_hooks_outputs(model): return Hooks(model, lambda model, input, output: output)

def get_model_outputs(model, size=(16, 16, 16)):
    hooks = get_hooks_outputs(model)
    ch = first_ch_sz(model)
    dummy = model.eval()(torch.zeros(1, ch, *size))
    outputs = hooks.outputs
    outputs_size = [hook.output.size() for hook in hooks]
    return outputs_size, dummy, outputs

def btnk(ch_i, ch_o): return nn.Conv3d(ch_i, ch_o, 1)

def CBR(ch_i, ch_o, ks=3, s=1, p=1):
    return nn.Sequential(nn.Conv3d(ch_i, ch_o, ks, s, p), nn.BatchNorm3d(ch_o), nn.ReLU())

def down(): return nn.MaxPool3d(2, 2)

def up(ch_i, ch_o, skp_f_sz, up_f_sz):
    # skp_f_sz, up_f_sz = (tuple(skp_f_sz), tuple(up_f_sz))
    s = skp_f_sz//up_f_sz
    ks = s
    op = skp_f_sz - (up_f_sz - 1)*s - ks
    return nn.ConvTranspose3d(ch_i, ch_o, tuple(ks), tuple(s), output_padding=tuple(op), bias=False)


class UnetBlock(nn.Module):

    def __init__(self, skp_opt, up_ch, skp_f_sz, up_f_sz):
        super().__init__()
        skp_ch = skp_opt.size(1)
        self.up = up(up_ch, skp_ch, skp_f_sz, up_f_sz)
        ch = 2*skp_ch
        self.CBR_1 = CBR(ch, ch//2)
        ch = ch//2
        self.CBR_2 = CBR(ch, ch)
        self.skp_opt = skp_opt

    def forward(self, x):
        x = self.up(x)
        x = torch.cat([self.skp_opt, x], dim=1)
        return self.CBR_2(self.CBR_1(x))


class AutoUnet(nn.Sequential):

    def __init__(self, encoder, n_cls=2, size=(16,32,16)):
        skip_outputs_size, dummy, skip_outputs = get_model_outputs(encoder, size)
        skip_sz_diff_idxs = _szdiff_idxs(skip_outputs_size)
        skip_sz_diff_idxs.reverse()

        ch = dummy.size(1)
        center = nn.Sequential(CBR(ch, ch), CBR(ch, 2*ch))
        dummy = center(dummy)
        layers = [encoder, center]
        dummy_szs = ['***center***', dummy.size(), '***center***']

        for idx in skip_sz_diff_idxs:
            skp_opt, up_ch, skp_f_sz, up_f_sz = \
                skip_outputs[idx], dummy.size(1), np.array(skip_outputs_size[idx][-3:]), np.array(dummy.shape[-3:])
            unet_block = UnetBlock(skp_opt, up_ch, skp_f_sz, up_f_sz)
            layers.append(unet_block)
            dummy = unet_block(dummy)

            dummy_szs.append(dummy.size())
            dummy_szs.insert(0, skip_outputs_size[idx])

        n = unet_block.CBR_2[0].out_channels
        layers.append(btnk(n, n_cls))

        super().__init__(*layers)
        self.dummy_size = dummy_szs












