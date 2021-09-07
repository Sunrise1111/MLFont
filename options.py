import argparse

def get_opt():
    parse = argparse.ArgumentParser()
    parse.add_argument('--metatrain_path', type=str, default=r'D:\1 - source\MLFont\MLFont-pytorch\fontset\train', help='meta-training set filepath')
    parse.add_argument('--finetune_path', type=str, default='', help='new font set filepath')
    parse.add_argument('--total_epoch', type=int, default=100, help='the number of epochs for fine-tuning stage')
    parse.add_argument('--test_per_epochs', type=int, default=3, help='test per epoch in the fine-tuning stage')
    parse.add_argument('--input_nc', type=int, default=3, help='input image channels')
    parse.add_argument('--output_nc', type=int, default=3, help='output image channels')
    parse.add_argument('--ngf', type=int, default=64, help='the number of generator filters in the last conv layer')
    parse.add_argument('--ndf', type=int, default=64, help='the number of discriminator filters in the first conv layer')
    parse.add_argument('--norm', type=str, default='batch', help='normalization layer [instance | batch | none]')
    parse.add_argument('--nl', type=str, default='relu', help='activation function [relu | lrelu]')
    parse.add_argument('--use_dropout', type=bool, default=False, help='use dropout')
    parse.add_argument('--init_type', type=str, default='xavier', help='parameters initialization [normal | xavier | kaiming | orthogonal]')
    parse.add_argument('--upsample', type=str, default='bilinear', help='the type of upsample layer')
    parse.add_argument('--gan_mode', type=str, default='lsgan',help='the type of GAN objective. [vanilla| lsgan | wgangp]')
    parse.add_argument('--beta1', type=float, default=0.5, help='momentum term of adam')
    parse.add_argument('--update_lr', type=float, default=0.0001, help='inner learning rate')
    parse.add_argument('--meta_lr', type=float, default=0.0001, help='outer learning rate')
    parse.add_argument('--n_font', type=int, default=12, help='the number of fonts in the meta-training set')
    parse.add_argument('--k_shot', type=int, default=2, help='the size of support set')
    parse.add_argument('--k_query', type=int, default=1, help='the size of query set, where k_query refer to the multiple of support set')
    parse.add_argument('--n_task', type=int, default=3, help='one meta-optimization use n tasks')
    parse.add_argument('--n_style', type=int, default=7, help='the number of target style image for one forward pass in the network')
    parse.add_argument('--m', type=int, default=1, help='the number of inner update steps')
    parse.add_argument('--lambda_L1', type=int, default=10, help='the weight of L1 loss in the fine-tuning stage')
    opt = parse.parse_args()
    return opt