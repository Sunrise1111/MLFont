import os
import time
import torch
import random
import torch.nn
from options import get_opt
from finetuningdata import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
from networks import define_G, define_D, GANLoss, set_requires_grad
from tensorboardX import SummaryWriter

def main():
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    path = './checksave/' + now
    opt = get_opt()
    print(opt)
    opt_dict = opt.__dict__
    if not os.path.exists(path):
        os.mkdir(path)
    if not os.path.exists(path + '/test_result'):
        os.mkdir(path + '/test_result')
    if not os.path.exists(path + '/train_result'):
        os.mkdir(path + '/train_result')

    with open(path + '/argparse.txt', 'w') as f:
        f.writelines('argparse:\n')
        for each_opt, value in opt_dict.items():
            f.writelines(each_opt + ' : ' + str(value) + '\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')

    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_style, opt.norm, opt.nl,opt.use_dropout, opt.init_type, opt.upsample)
    netD = define_D(opt.input_nc + opt.output_nc, opt.ndf, opt.norm, opt.init_type)

    train_dataset = Dataset(opt.finetune_path, model='train', style=opt.n_style)
    train_loader = DataLoader(dataset=train_dataset, batch_size=4, num_workers=0, pin_memory=True)

    test_dataset = Dataset(opt.finetune_path, model='test', style=opt.n_style)
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, pin_memory=True)

    optimizer_G = torch.optim.Adam(netG.parameters(), lr=opt.update_lr, betas=(opt.beta1, 0.999))
    optimizer_D = torch.optim.Adam(netD.parameters(), lr=opt.update_lr, betas=(opt.beta1, 0.999))
    criterionGAN = GANLoss(opt.gan_mode).to(device)

    writer = SummaryWriter()
    total_iters = 0
    test_iter = 0

    netG.load_state_dict(torch.load('./pre_model/G.pth'))
    for epoch in range(opt.total_epoch):

        netG.train()
        netD.train()
        for i,batch in enumerate(train_loader):
            a, style, b = batch
            a = a.to(device)
            style = style.to(device)
            b = b.to(device)
            fake_b = netG(a, style)

            save_image(fake_b, path + '/train_result/epoch{}_{}.png'.format(epoch, i), padding=20, pad_value=1)

            set_requires_grad(netD, True)
            optimizer_D.zero_grad()
            fake_ab = torch.cat((a, fake_b), 1)
            pred_fake = netD(fake_ab.detach())

            loss_d_fake = criterionGAN(pred_fake, False)
            real_ab = torch.cat((a, b), 1)
            pred_real = netD(real_ab)
            loss_d_real = criterionGAN(pred_real, True)
            loss_d = (loss_d_fake + loss_d_real) * 0.5
            loss_d.backward()
            optimizer_D.step()

            set_requires_grad(netD, False)
            optimizer_G.zero_grad()
            fake_ab = torch.cat((a, fake_b), 1)
            pred_fake = netD(fake_ab)
            loss_G_GAN = criterionGAN(pred_fake, True)
            l1 = torch.nn.L1Loss()(fake_b, b)
            loss_G_l1 = l1 * opt.lambda_L1

            loss_g = loss_G_GAN + loss_G_l1

            with open(path + '/train_process.txt', 'a') as f:
                f.writelines('epoch   {}   step   {}   d_real   {:.6f}   d_fake   {:.6f}   l1loss   {:.6f}   d_generate : {:.6f}\n'.format(epoch, i, loss_d_real.float(), loss_d_fake.float(), l1.float(), loss_G_GAN.float()))
            print('epoch   {}   batch   {}   d_real   {:.6f}   d_fake   {:.6f}   l1loss   {:.6f}   d_generate : {:.6f}\n'.format(epoch, i, loss_d_real.float(), loss_d_fake.float(), l1.float(), loss_G_GAN.float()))

            d = {}
            d['d_real'] = loss_d_real
            d['d_fake'] = loss_d_fake
            d['l1'] = l1
            d['d_generate'] = loss_G_GAN
            writer.add_scalars('train_process', d, total_iters)
            total_iters += 1

            loss_g.backward()
            optimizer_G.step()

        torch.save(netG.state_dict(), path + '/netG.pth')
        torch.save(netD.state_dict(), path + '/netD.pth')

        if epoch % opt.test_per_epochs == 0:
            with torch.no_grad():
                netG.eval()
                total_l1 = 0
                for i, test in enumerate(test_loader):
                    index = random.randint(0, len(train_dataset) - 1)
                    _, style, _ = train_dataset[index]
                    style = style.unsqueeze(0)
                    a, _, b = test
                    a = a.to(device)
                    style = style.to(device)
                    b = b.to(device)
                    fake_b = netG(a, style)
                    test_l1 = torch.nn.L1Loss()(fake_b, b)
                    total_l1 += test_l1
                    save_image(fake_b, path + '/test_result/epoch{}_{}.png'.format(epoch, i), padding=20, pad_value=1)
                    with open(path + '/test_loss.txt', 'a') as f:
                        f.writelines('epoch   {}   batch   {}   l1_loss : {}\n'.format(epoch, i, test_l1))

                    writer.add_scalar('test_l1', test_l1, test_iter)
                    test_iter += 1
                avg_l1 = total_l1/64
                print(avg_l1)

if __name__ == '__main__':
    main()