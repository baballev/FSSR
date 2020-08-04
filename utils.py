import imghdr
import os
from PIL import Image
import os
import torch
import torchvision
import torchvision.transforms as transforms
import matplotlib.pyplot as plt

os.chdir('E:/Programmation/Python/HArchiver/')

def show_image(input_tensor, n=0):
    y = input_tensor.detach()[n].cpu().numpy().transpose((1, 2, 0))
    plt.imshow(y)
    plt.pause(1)


## Utils
def isFileNotCorrupted(path):
    return (imghdr.what(path) == 'jpeg' or imghdr.what(path) == 'png') # Checks the first bytes of the file to see if it's a valid png/jpeg

class DADataset(torch.utils.data.Dataset): # Making artificial tasks with Data Augmentation.
    def __init__(self, images_directory, num_shot, transform, is_valid_file=isFileNotCorrupted, scale_factor=2):
        self.is_valid_file = is_valid_file
        self.image_paths = [os.path.join(images_directory, f) for f in os.listdir(images_directory) if self.is_valid_file(os.path.join(images_directory, f))]
        self.length = len(self.image_paths)
        self.transform = transform
        self.scale_factor = scale_factor
        self.num_shot = num_shot

    def __getitem__(self, index):

        original = Image.open(self.image_paths[index]).convert('RGB')
        width, height = original.width, original.height

        support_label, support_data = [], []
        augmentation = transforms.Compose([transforms.ColorJitter(brightness=0.5, contrast=0.5, saturation=0.5, hue=(-0.1,0.1)), transforms.RandomPerspective(), transforms.RandomRotation((-15,15)), transforms.RandomResizedCrop((height//self.scale_factor, width//self.scale_factor), scale=(0.5, 0.95), interpolation=Image.BICUBIC), transforms.RandomGrayscale(p=0.02), transforms.RandomHorizontalFlip(0.3), transforms.RandomVerticalFlip(0.3)])
        for i in range(self.num_shot):
            transformed_img = augmentation(original)
            support_label.append(self.transform(transformed_img))
            support_data.append(self.transform(transforms.Resize((height//(self.scale_factor*2), width//(self.scale_factor*2)), interpolation=Image.BICUBIC)(transformed_img)))

        query_data, query_label = self.transform(transforms.Resize((height//self.scale_factor, width//self.scale_factor))(original)), self.transform(original) # ToDo: Change code to make the set more customizable?
        # Todo: change list to tensor so that we can batch it? -> no we can't because images are not of the same size.
        return support_data, torch.stack(support_label), query_data, query_label
        # support_data is a list !

    def __len__(self):
        return self.length


class FSDataset(torch.utils.data.Dataset):
    ''' Assuming classes_folder_path is a directory containing folders of N images of the same category,
        We take N-1 images for the support set and 1 image for the query set.
        The support set is composed of N-1 couples of images (downsampled_image, original_image).
        The query set is composed of 1 image couple (downsampled_image, original image).
        In this setup, it is a N-1 shot (1 way) super-resolution task.
    '''
    def __init__(self, classes_folder_path, transform, is_valid_file, color_mode="RGB", scale_factor=2): # color_mode = "RGB" or "YCbCr" or "Y"
        self.is_valid_file = is_valid_file
        self.class_paths = [os.path.join(classes_folder_path, f) for f in os.listdir(classes_folder_path) if os.path.isdir(f)]
        self.length = len(self.class_paths)
        self.transform = transform
        self.color_mode = color_mode
        self.scale_factor = scale_factor

    def __getitem__(self, index): # ToDO
        image_path = self.class_paths[index]
        if self.color_mode == 'Y':
            imageHR = Image.open(image_path).convert('YCbCr').getchannel(0)
        else:
            imageHR = Image.open(image_path).convert(self.color_mode)
        width, height = imageHR.width, imageHR.height
        imageLR = transforms.Compose([transforms.Resize((height//self.scale_factor, width//self.scale_factor))] + self.transform.transforms)(imageHR)
        imageHR = self.transform(imageHR)
        return imageLR, imageHR

    def __len__(self):
        return self.length




