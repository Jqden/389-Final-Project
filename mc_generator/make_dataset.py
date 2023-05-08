import torch
import os
from torchvision import datasets, transforms
from PIL import ImageDraw, Image
import matplotlib.pyplot as plt
import numpy as np
import random
from tqdm import tqdm


folder_path = r"C:\Users\jaden\OneDrive\Desktop\CS 389\FinalProj\skins"
output_path = r"C:\Users\jaden\OneDrive\Desktop\CS 389\FinalProj\skins_faces"

# for filename in tqdm(os.listdir(folder_path)):
#     try:
#         image = Image.open(os.path.join(folder_path, filename))
#         image = image.convert("RGB")
#         image = image.crop((0, 0, 32, 32))

#         draw = ImageDraw.Draw(image)
#         draw.rectangle((0, 0, 7, 7), fill=(255, 255, 255))
#         draw.rectangle((image.width-8, 0, image.width-1, 7), fill=(255, 255, 255))
#         draw.rectangle((0, 16, image.width-1, image.height-1), fill=(255, 255, 255))

#         img_name = os.path.splitext(filename)[0] + '.jpg'
#         img_path = os.path.join(output_path, img_name)
#         image.save(img_path, quality=100, subsampling=0)

#     except:
#         print(filename, "was bad")


class CustomDataset(torch.utils.data.Dataset):
    def __init__(self, root_dir, transform=None):
        self.root_dir = root_dir
        self.transform = transform
        self.image_filenames = os.listdir(root_dir)
        
    def __len__(self):
        return len(self.image_filenames)
    
    def __getitem__(self, index):
        # Load the image
        image = Image.open(os.path.join(self.root_dir, self.image_filenames[index]))
        
        preprocess = transforms.Compose(
            [
                transforms.Resize((32, 32)),
                transforms.RandomHorizontalFlip(),
                transforms.ToTensor(),
                transforms.Normalize([0.5], [0.5]),
            ]
        )
        # Apply any additional transforms
        image = preprocess(image)
        
        return image


dataset = CustomDataset(r"C:\Users\jaden\OneDrive\Desktop\CS 389\FinalProj\skins_faces")
x = dataset[0].unsqueeze(0)
print(x.shape)