import torchvision.transforms as transforms
import torch
import os
import sys
from PIL import Image
"""
os.chdir(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
print(os.curdir)
"""
#sys.path.append("..")
#from utils import DADataset


output_path = ''
input_path = ''

shot_num = 10

validset = DADataset(input_path, shot_num, transform=transforms.ToTensor())
validloader = torch.utils.data.DataLoader(validset, batch_size=1, num_workers=0, shuffle=False)

img_transform = transforms.ToPILImage()

for i, data in enumerate(validloader):
    os.mkdir(os.path.join(output_path, str(i)))
    spt_label, q_label = data[1], data[3]
    for k in range(shot_num):
        img = img_transform(spt_label[:, k, :, :, :][0])
        img.save(os.path.join(os.path.join(output_path, str(i)), str(k) + '.png'))
    img = img_transform(q_label[0])
    img.save(os.path.join(os.path.join(output_path, str(i)), str(shot_num) + '.png'))

