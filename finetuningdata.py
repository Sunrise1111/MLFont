import os
import torch
import random
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

    def __init__(self, data_path, model='train', style=7):
        self.style = style
        A_path = os.path.join(data_path, model+'/A')
        B_path = os.path.join(data_path, model+'/B')
        list_A = sorted([os.path.join(A_path,i) for i in os.listdir(A_path)])
        self.list_A = list_A
        list_B = sorted([os.path.join(B_path,i) for i in os.listdir(B_path)])
        self.list_B = list_B

    def __getitem__(self, index):
        A_path = self.list_A[index]
        img_a = Image.open(A_path).convert('RGB')
        a = get_transform()(img_a)

        B_path = self.list_B[index]
        img_b = Image.open(B_path).convert('RGB')
        b = get_transform()(img_b)

        list_b = random.sample(self.list_B, self.style)
        list_img_b = [Image.open(i).convert('RGB') for i in list_b]
        transform_b = list(map(lambda c: transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))(transforms.ToTensor()(c)), list_img_b))
        style = torch.cat(transform_b)

        return a, style, b

    def __len__(self):
        return len(self.list_A)