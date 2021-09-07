import os
import time
import torch
import torch.nn as nn
from metadata import Dataset
from options import get_opt
from networks import define_G
from tensorboardX import SummaryWriter
from util import make_functional

def main():
    now = time.strftime("%Y-%m-%d-%H_%M_%S", time.localtime())
    path = './checksave/' + now
    opt = get_opt()
    print(opt)
    opt_dict = opt.__dict__
    if not os.path.exists(path):
        os.makedirs(path)
    if not os.path.exists(path + '/saved_model'):
        os.mkdir(path + '/saved_model')
    if not os.path.exists(path + '/train_result'):
        os.mkdir(path + '/train_result')
    with open(path + '/argparse.txt', 'w') as f:
        f.writelines('argparse:\n')
        for each_opt, value in opt_dict.items():
            f.writelines(each_opt + ' : ' + str(value) + '\n')

    device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
    netG = define_G(opt.input_nc, opt.output_nc, opt.ngf, opt.n_style, opt.norm, opt.nl, opt.use_dropout, opt.init_type, opt.upsample)
    meta_optimizer = torch.optim.Adam(netG.parameters(), lr=opt.meta_lr, betas=(opt.beta1, 0.999))
    train_dataset = Dataset(opt.metatrain_path, n_font=opt.n_font, k_shot=opt.k_shot, k_query=opt.k_query, n_task=opt.n_task, n_style=opt.n_style)

    netG.train()
    total_iters = 0
    writer = SummaryWriter()
    for epoch in range(opt.total_epoch):

        for i,batch in enumerate(train_dataset):

            a_list, style_list, b_list = batch

            loss_query = 0
            for j, (a, style, b) in enumerate(zip(a_list, style_list, b_list)):

                style = style.to(device)
                style = style.unsqueeze(0)
                support_style = torch.cat([style for i in range(opt.k_shot)])

                support_a, query_a = a
                support_a = support_a.to(device)
                query_a = query_a.to(device)

                support_b, query_b = b
                support_b = support_b.to(device)
                query_b = query_b.to(device)

                f_netG = make_functional(netG)
                fast_weights = list(netG.parameters())
                for _ in range(opt.m):
                    fake_b = f_netG(support_a, support_style, params=fast_weights)
                    support_l1 = nn.L1Loss()(fake_b, support_b)
                    grad = torch.autograd.grad(support_l1, fast_weights, create_graph=True)
                    fast_weights = list(map(lambda p: p[1] - opt.update_lr * p[0], zip(grad, netG.parameters())))

                query_style = torch.cat([style for i in range(opt.k_query*opt.k_shot)])
                query_fake_b = f_netG(query_a, query_style, params=fast_weights)
                query_l1 = nn.L1Loss()(query_fake_b, query_b)
                loss_query = query_l1 if loss_query == 0 else loss_query + query_l1

                if i%3==0:
                    with open(path + '/train_process.txt', 'a') as f:
                        f.writelines('epoch   {}   batch  {}   task   {}   support l1 loss   {:.6f}   query l1 loss   {:.6f}\n'.format(epoch, i, j, support_l1.float(), query_l1.float()))
                    print('epoch   {}   batch   {}   task   {}   support l1 loss   {:.6f}   query l1 loss   {:.6f}\n'.format(epoch, i, j, support_l1.float(), query_l1.float()))
                d = {}
                d['support_l1'] = support_l1
                d['query_l1'] = query_l1
                writer.add_scalars('train_process', d, total_iters)
                total_iters += 1

            meta_optimizer.zero_grad()
            loss_query.backward()
            meta_optimizer.step()

            if i>=1352:
                break

        if epoch%500==0:
            torch.save(netG.state_dict(), path + '/saved_model/{}G.pth'.format(epoch))

        torch.save(netG.state_dict(), path + '/saved_model/G.pth')

if __name__ == '__main__':
    main()