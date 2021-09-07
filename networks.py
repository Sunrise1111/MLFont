import torch
import functools
import torch.nn as nn
from torch.nn import init

def upsampleLayer(inplanes, outplanes, upsample='basic'):
    if upsample == 'basic':
        upconv = [nn.ConvTranspose2d(inplanes, outplanes, kernel_size=4, stride=2, padding=1)]
    elif upsample == 'bilinear':
        upconv = [nn.Upsample(scale_factor=2, mode='bilinear', align_corners=True),
                  nn.ReflectionPad2d(1)]
        upconv += [nn.Conv2d(inplanes, outplanes, kernel_size=3, stride=1, padding=0)]
    else:
        raise NotImplementedError("upsample layer {} not implemented".format(upsample))
    return upconv

class MLFontBlock(nn.Module):
    def __init__(self, input_cont, input_style, outer_nc, inner_nc, submodule=None, outermost=False, innermost=False, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', wo_skip=False):
        super(MLFontBlock, self).__init__()
        self.wo_skip = wo_skip
        downconv1 = []
        downconv2 = []
        self.outermost = outermost
        self.innermost = innermost
        downconv1 += [nn.Conv2d(input_cont, inner_nc, kernel_size=3, stride=2, padding=1)]
        downconv2 += [nn.Conv2d(input_style, inner_nc, kernel_size=3, stride=2, padding=1)]
        downrelu1 = nn.LeakyReLU(0.2, True)
        downrelu2 = nn.LeakyReLU(0.2, True)
        uprelu2 = nl_layer()

        if outermost:
            if self.wo_skip:
                upconv_B = upsampleLayer(inner_nc, outer_nc, upsample=upsample)
            else:
                upconv_B = upsampleLayer(inner_nc * 3, outer_nc, upsample=upsample)
            down1 = downconv1
            down2 = downconv2
            up_B = [uprelu2] + upconv_B + [nn.Tanh()]

        elif innermost:
            upconv_B = upsampleLayer(inner_nc * 2, outer_nc, upsample=upsample)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            up_B = [uprelu2] + upconv_B
            if norm_layer is not None:
                up_B += [norm_layer(outer_nc)]
        else:
            if self.wo_skip:
                upconv_B = upsampleLayer(inner_nc, outer_nc, upsample=upsample)
            else:
                upconv_B = upsampleLayer(inner_nc * 3, outer_nc, upsample=upsample)
            down1 = [downrelu1] + downconv1
            down2 = [downrelu2] + downconv2
            if norm_layer is not None:
                down1 += [norm_layer(inner_nc)]
                down2 += [norm_layer(inner_nc)]
            up_B = [uprelu2] + upconv_B
            if norm_layer is not None:
                up_B += [norm_layer(outer_nc)]
            if use_dropout:
                up_B += [nn.Dropout(0.5)]

        self.down1 = nn.Sequential(*down1)
        self.down2 = nn.Sequential(*down2)
        self.submodule = submodule
        self.up_B = nn.Sequential(*up_B)

    def forward(self, content, style):
        x1 = self.down1(content)
        x2 = self.down2(style)
        if self.outermost:
            mid_B = self.submodule(x1, x2)
            fake_B = self.up_B(mid_B)
            return fake_B
        elif self.innermost:

            mid_B = torch.cat([x1, x2], 1)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_B
            else:
                return torch.cat([fake_B, tmp1], 1)
        else:
            mid_B = self.submodule(x1, x2)
            fake_B = self.up_B(mid_B)
            tmp1 = torch.cat([content, style], 1)
            if self.wo_skip:
                return fake_B
            else:
                return torch.cat([fake_B, tmp1], 1)

class MLFont(nn.Module):
    def __init__(self, input_content, input_style, output_nc, num_downs, ngf=64, norm_layer=None, nl_layer=None, use_dropout=False, upsample='basic', wo_skip=False):
        super(MLFont, self).__init__()
        max_nchn = 8
        block = MLFontBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, innermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        for i in range(num_downs - 5):
            block = MLFontBlock(ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, ngf*max_nchn, block, norm_layer=norm_layer, nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample, wo_skip=wo_skip)
        block = MLFontBlock(ngf*4, ngf*4, ngf*4, ngf*max_nchn, block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        block = MLFontBlock(ngf*2, ngf*2, ngf*2, ngf*4, block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        block = MLFontBlock(ngf, ngf, ngf, ngf*2, block, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        block = MLFontBlock(input_content, input_style, output_nc, ngf, block, outermost=True, norm_layer=norm_layer, nl_layer=nl_layer, upsample=upsample, wo_skip=wo_skip)
        self.model = block

    def forward(self, content, style):
        return self.model(content, style)

class NLayerDiscriminator(nn.Module):
    def __init__(self, input_nc, ndf=64, n_layers=3, norm_layer=nn.BatchNorm2d):
        super(NLayerDiscriminator, self).__init__()
        if type(norm_layer) == functools.partial:  # no need to use bias as BatchNorm2d has affine parameters
            use_bias = norm_layer.func == nn.InstanceNorm2d
        else:
            use_bias = norm_layer == nn.InstanceNorm2d
        kw = 4
        padw = 1
        sequence = [nn.Conv2d(input_nc, ndf, kernel_size=kw, stride=2, padding=padw), nn.LeakyReLU(0.2, True)]
        nf_mult = 1

        for n in range(1, n_layers):
            nf_mult_prev = nf_mult
            nf_mult = min(2 ** n, 8)
            sequence += [
                nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=2, padding=padw, bias=use_bias),
                norm_layer(ndf * nf_mult),
                nn.LeakyReLU(0.2, False)
            ]

        nf_mult_prev = nf_mult
        nf_mult = min(2 ** n_layers, 8)
        sequence += [
            nn.Conv2d(ndf * nf_mult_prev, ndf * nf_mult, kernel_size=kw, stride=1, padding=padw, bias=use_bias),
            norm_layer(ndf * nf_mult),
            nn.LeakyReLU(0.2, True)
        ]

        sequence += [nn.Conv2d(ndf * nf_mult, 1, kernel_size=kw, stride=1, padding=padw)]  # output 1 channel prediction map
        self.model = nn.Sequential(*sequence)

    def forward(self, input):
        return self.model(input)

def init_weights(net, init_type='normal', gain=0.02):
    def init_func(m):
        classname = m.__class__.__name__
        if hasattr(m, 'weight') and (classname.find('Conv') != -1 or classname.find('Linear') != -1):
            if init_type == 'normal':
                init.normal_(m.weight.data, 0.0, gain)
            elif init_type == 'xavier':
                init.xavier_normal_(m.weight.data, gain=gain)
            elif init_type == 'kaiming':
                init.kaiming_normal_(m.weight.data, a=0, mode='fan_in')
            elif init_type == 'orthogonal':
                init.orthogonal_(m.weight.data, gain=gain)
            else:
                raise NotImplementedError('initialization method [%s] is not implemented' % init_type)
            if hasattr(m, 'bias') and m.bias is not None:
                init.constant_(m.bias.data, 0.0)
        elif classname.find('BatchNorm2d') != -1:
            init.normal_(m.weight.data, 1.0, gain)
            init.constant_(m.bias.data, 0.0)

    print('initialize network with %s' % init_type)
    net.apply(init_func)

def init_net(net, init_type='normal'):
    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    net.to(device)
    init_weights(net, init_type)
    return net

def get_norm_layer(norm_type='batch'):
    if norm_type == 'batch':
        norm_layer = functools.partial(nn.BatchNorm2d, affine=True)
    elif norm_type == 'instance':
        norm_layer = functools.partial(nn.InstanceNorm2d, affine=False)
    elif norm_type == 'none':
        norm_layer = None
    else:
        raise NotImplementedError(
            'normalization layer [%s] is not found' % norm_type)
    return norm_layer

def get_non_linearity(layer_type='relu'):
    if layer_type == 'relu':
        nl_layer = functools.partial(nn.ReLU, inplace=False)
    elif layer_type == 'lrelu':
        nl_layer = functools.partial(nn.LeakyReLU, negative_slope=0.2, inplace=False)
    else:
        raise NotImplementedError('nonlinearity activitation [%s] is not found' % layer_type)
    return nl_layer

def define_G(input_nc, output_nc, ngf, nencode=4, norm='batch', nl='relu', use_dropout=False, init_type='xavier', upsample='bilinear'):
    norm_layer = get_norm_layer(norm_type=norm)
    nl_layer = get_non_linearity(layer_type=nl)
    input_content = input_nc
    input_style = input_nc * nencode
    net = MLFont(input_content, input_style, output_nc, 6, ngf, norm_layer=norm_layer,  nl_layer=nl_layer, use_dropout=use_dropout, upsample=upsample)
    return init_net(net, init_type)

def define_D(input_nc, ndf, norm='batch', init_type='normal'):
    norm_layer = get_norm_layer(norm_type=norm)
    net = NLayerDiscriminator(input_nc, ndf, n_layers=3, norm_layer=norm_layer)
    return init_net(net, init_type)

class GANLoss(nn.Module):
    def __init__(self, gan_mode, target_real_label=1.0, target_fake_label=0.0):
        super(GANLoss, self).__init__()
        self.register_buffer('real_label', torch.tensor(target_real_label))
        self.register_buffer('fake_label', torch.tensor(target_fake_label))
        self.gan_mode = gan_mode
        if gan_mode == 'lsgan':
            self.loss = nn.MSELoss()
        elif gan_mode == 'vanilla':
            self.loss = nn.BCEWithLogitsLoss()
        elif gan_mode in ['wgangp']:
            self.loss = None
        else:
            raise NotImplementedError('gan mode %s not implemented' % gan_mode)

    def get_target_tensor(self, prediction, target_is_real):
        if target_is_real:
            target_tensor = self.real_label
        else:
            target_tensor = self.fake_label
        return target_tensor.expand_as(prediction)

    def __call__(self, prediction, target_is_real):
        loss = None
        if self.gan_mode in ['lsgan', 'vanilla']:
            target_tensor = self.get_target_tensor(prediction, target_is_real)
            loss = self.loss(prediction, target_tensor)
        elif self.gan_mode == 'wgangp':
            if target_is_real:
                loss = -prediction.mean()
            else:
                loss = prediction.mean()
        return loss

def set_requires_grad(nets, requires_grad=False):
    if not isinstance(nets, list):
        nets = [nets]
    for net in nets:
        if net is not None:
            for param in net.parameters():
                param.requires_grad = requires_grad