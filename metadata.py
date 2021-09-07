import os
import torch
import random
import numpy as np
from PIL import Image
import torch.utils.data as data
import torchvision.transforms as transforms

def get_transform(gray = False):
    transform_list = []
    transform_list += [transforms.ToTensor()]
    if gray:
        transform_list += [transforms.Normalize((0.5,), (0.5,))]
    else:
        transform_list += [transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))]
    return transforms.Compose(transform_list)

class Dataset(data.Dataset):

    def __init__(self, data_path, n_font, k_shot, k_query, n_task, n_style):
        self.n_font = n_font
        self.k_shot = k_shot
        self.k_query = k_query
        self.n_task = n_task
        self.n_style = n_style
        list_B = sorted([os.path.join(data_path, i) for i in os.listdir(data_path)])
        self.list_B_dir = list_B
        self.list_A = sorted(os.listdir(list_B[0]))

    def __getitem__(self, index):

        sample_A = random.sample(self.list_A, self.k_shot + self.k_shot*self.k_query)

        total = [i for i in range(self.n_font)]
        i_font = random.sample(total, self.n_task)
        content = random.sample(list(set(total)-set(i_font)), self.n_task)
        a_list = []
        style_list = []
        b_list = []
        for c, j in zip(content, i_font):

            img_A = [Image.open(os.path.join(self.list_B_dir[c], i)).convert('RGB') for i in sample_A]
            img_A = list(map(lambda c: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), img_A))
            support_A = img_A[:self.k_shot]
            query_A = img_A[self.k_shot:]
            a = [support_A, query_A]
            a_list.append(a)

            b_dir = self.list_B_dir[j]
            list_B = sorted(os.listdir(b_dir))
            style = random.sample(list_B, self.n_style)
            style_img = [Image.open(os.path.join(b_dir, i)).convert('RGB') for i in style]
            style_img = list(map(lambda c: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), style_img))
            style = torch.cat(style_img)
            style_list.append(style)

            img_B = [Image.open(os.path.join(b_dir, i)).convert('RGB') for i in sample_A]
            img_B = list(map(lambda c: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), img_B))
            support_B = img_B[:self.k_shot]
            query_B = img_B[self.k_shot:]
            b = [support_B, query_B]
            b_list.append(b)

        t1 = []
        for p in a_list:
            r1 = []
            for i in p:
                r = []
                for s in i:
                    r.append(s.numpy())
                r1.append(torch.from_numpy(np.array(r)))
            t1.append(r1)
        a_list = t1

        o = []
        for p in b_list:
            r1 = []
            for i in p:
                r = []
                for s in i:
                    r.append(s.numpy())
                r1.append(torch.from_numpy(np.array(r)))
            o.append(r1)
        b_list = o

        return a_list, style_list, b_list

    def __len__(self):
        return len(self.list_A)//self.k_shot