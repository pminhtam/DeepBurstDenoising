import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from metric import  psnr
import torchvision.transforms as transforms
import glob
from PIL import Image

def load_data(dir,image_size):
    image_files = sorted(glob.glob(dir + "/*"))[:8]
    print(image_files)
    image_0 = Image.open(image_files[0]).convert('RGB')
    w = image_size
    h = image_size
    # print(image_0.size[-1])
    nw = image_0.size[0] - w
    nh = image_0.size[1] - h
    print(nw,nh)
    if nw < 0 or nh < 0:
        raise RuntimeError("Image is to small {} for the desired size {}". \
                           format((image_0.size(-1), image_0.size(-2)), (w, h))
                           )
    idx_w = torch.randint(0, nw + 1, (1,))[0]
    idx_h = torch.randint(0, nh + 1, (1,))[0]
    print(idx_w,idx_h)
    image_noise = [transforms.ToTensor()(Image.open(img_path).convert('RGB'))[:, idx_h:(idx_h + h), idx_w:(idx_w + w)] for
               img_path in image_files]
    image_noise_burst_crop = torch.stack(image_noise, dim=0)
    image_noise_burst_crop = image_noise_burst_crop.unsqueeze(0)
    print("image_noise_burst_crop shape : ",image_noise_burst_crop.size())
    return image_noise_burst_crop

def test_multi(dir,image_size,checkpoint,resume):
    num_workers = 1
    batch_size = 1
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model_single = SFD_C().to(device)
    model = MFD_C(model_single).to(device)
    if resume != '':
        print(device)
        save_dict = torch.load(os.path.join(checkpoint, resume), map_location=torch.device('cpu'))

        # if device == "cpu":
        #     save_dict = torch.load(os.path.join(checkpoint,resume),map_location=torch.device('cpu'))
        # else:
        #     save_dict = torch.load(os.path.join(checkpoint, resume))
        model.load_state_dict(save_dict['state_dict'])
        # model= save_dict['state_dict']
    trans = transforms.ToPILImage()
    model.eval()
    for i in range(10):
        image_noise = load_data(dir,image_size)
        image_noise_batch = image_noise.to(device)
        print(image_noise_batch.size())
        batch_size_i = image_noise_batch.size()[0]
        mfinit1, mfinit2, mfinit3,mfinit4,mfinit5,mfinit6,mfinit7 = torch.zeros(7, 1, 64, image_size, image_size).to(device)
        mfinit8 = torch.zeros(1, 3, image_size, image_size).to(device)
        i = 0
        for i_burst in range(batch_size_i):
            frame = image_noise_batch[:,i_burst,:,:,:]
            print(frame.size())
            if i == 0:
                i += 1
                dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8 = model(
                    frame, mfinit1, mfinit2, mfinit3, mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                plt.imshow(np.array(trans(dframe[0])))
                plt.title("denoise 0")
                plt.show()
            else:
                dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8= model(dframe, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8)
        # print(np.array(trans(mf8[0])))

        plt.imshow(np.array(trans(dframe[0])))
        plt.title("denoise")
        plt.show()
        plt.imshow(np.array(trans(image_noise[0][0])))
        plt.title("noise ")
        plt.show()

if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/samples/samples', help='path to noise image file')
    # parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--image_size','-sz' , type=int,default=128, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume', '-r', type=str, default="MFD_C_47.pth.tar",
                        help='file name of checkpoint')
    parser.add_argument('--type_model', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'multi':
        test_multi(args.noise_dir,args.image_size,args.checkpoint,args.resume)



