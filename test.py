import os
import time
import torch
import random
import torch.nn
from options import get_opt
from networks import define_G
from finetuningdata import Dataset
from torch.utils.data import DataLoader
from torchvision.utils import save_image
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

    with open(path + '/argparse.txt', 'w') as f:
        f.writelines('argparse:\n')
        for each_opt, value in opt_dict.items():
            f.writelines(each_opt + ' : ' + str(value) + '\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_style, opt.use_spectral_norm, opt.norm, opt.nl,opt.use_dropout, opt.use_attention, opt.init_type, opt.upsample)

    train_dataset = Dataset(opt.finetune_path, model='train')

    test_dataset = Dataset(opt.finetune_path, model='test')
    test_loader = DataLoader(dataset=test_dataset, batch_size=1, num_workers=0, pin_memory=True)

    writer = SummaryWriter()
    test_iter = 0

    netG.load_state_dict(torch.load(''))
    with torch.no_grad():
        netG.eval()
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

            save_image(fake_b, path + '/test_result/{}.png'.format(i), padding=20, pad_value=1)
            with open(path + '/test_loss.txt', 'a') as f:
                f.writelines('batch   {}   l1_loss : {}\n'.format(i, test_l1))

            writer.add_scalar('test_l1', test_l1, test_iter)
            test_iter += 1
            print(i)

if __name__ == '__main__':
    main()