import torch.utils.data as data
import torch
from PIL import Image
import os
import os.path
import glob
import torchvision.transforms as transforms
##

IMG_EXTENSIONS = [
    '.jpg', '.JPG', '.jpeg', '.JPEG',
    '.png', '.PNG', '.ppm', '.PPM', '.bmp', '.BMP',
]

class SingleLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir,gt_dir,image_size=512):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir +"/*" + files_ext))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir +"/*" + files_ext))
        
        if len(self.noise_path) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        

        self.transforms = transforms.Compose([transforms.Resize((self.image_size, self.image_size)),transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue 
        """
        image_noise = Image.open(self.noise_path[index]).convert('RGB')
        name_image = self.noise_path[index].split("/")[-1].replace("NOISY_","GT_")
        image_gt = Image.open(os.path.join(self.gt_dir, name_image)).convert('RGB')

        image_noise = self.transforms(image_noise)
        image_gt = self.transforms(image_gt)

        return image_noise, image_gt


    def __len__(self):
        return len(self.noise_path)

# --------------------------------------------
# inverse of pixel_shuffle
# --------------------------------------------
def pixel_unshuffle(input, upscale_factor):
    r"""Rearranges elements in a Tensor of shape :math:`(C, rH, rW)` to a
    tensor of shape :math:`(*, r^2C, H, W)`.

    Authors:
        Zhaoyi Yan, https://github.com/Zhaoyi-Yan
        Kai Zhang, https://github.com/cszn/FFDNet

    Date:
        01/Jan/2019
    """
    # print(input.size())
    channels, in_height, in_width = input.size()

    out_height = in_height // upscale_factor
    out_width = in_width // upscale_factor

    input_view = input.contiguous().view(
        channels, out_height, upscale_factor,
        out_width, upscale_factor)

    # channels *= upscale_factor ** 2
    unshuffle_out = input_view.permute(0, 2, 4, 1, 3).contiguous()
    return unshuffle_out.view(upscale_factor ** 2,channels, out_height, out_width)


class MultiLoader(data.Dataset):
    """
    Args:

     Attributes:
        noise_path (list):(image path)
    """

    def __init__(self, noise_dir, gt_dir, image_size=512):

        self.noise_dir = noise_dir
        self.gt_dir = gt_dir
        self.image_size = image_size
        self.noise_path = []
        for files_ext in IMG_EXTENSIONS:
            self.noise_path.extend(glob.glob(self.noise_dir + "/*" + files_ext))
        self.gt_path = []
        for files_ext in IMG_EXTENSIONS:
            self.gt_path.extend(glob.glob(self.gt_dir + "/*" + files_ext))

        if len(self.noise_path) == 0:
            raise (RuntimeError("Found 0 images in subfolders of: " + self.noise_dir + "\n"
                                                                                       "Supported image extensions are: " + ",".join(
                IMG_EXTENSIONS)))

        self.transforms_noise = transforms.Compose([transforms.Resize((2*self.image_size, 2*self.image_size)), transforms.ToTensor()])
        self.transforms_gt = transforms.Compose([transforms.Resize((self.image_size, self.image_size)), transforms.ToTensor()])

    def __getitem__(self, index):
        """
        Args:
            index (int): Index

        Returns:
            tuple: (image, groundtrue) where image is a noisy version of groundtrue
        """
        image_noise = Image.open(self.noise_path[index]).convert('RGB')
        name_image = self.noise_path[index].split("/")[-1].replace("NOISY_", "GT_")
        image_gt = Image.open(os.path.join(self.gt_dir, name_image)).convert('RGB')

        image_noise = self.transforms_noise(image_noise)
        image_noise_burst = pixel_unshuffle(image_noise,2)
        image_gt = self.transforms_gt(image_gt)

        return image_noise_burst, image_gt

    def __len__(self):
        return len(self.noise_path)

