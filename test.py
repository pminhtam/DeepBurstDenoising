import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim
import matplotlib.pyplot as plt
from metric import  psnr

def test_single(noise_dir,gt_dir,image_size,num_workers,checkpoint,resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    dataset = SingleLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=num_workers)
    model = SFD_C().to(device)
    if resume != '':
        save_dict = torch.load(os.path.join(checkpoint,resume))
        # model.load_state_dict(save_dict['state_dict'])
        model.load_state_dict(save_dict['state_dict'])
    for step, (image_noise, image_gt) in enumerate(data_loader):
        image_noise = image_noise.to(device)
        image_gt = image_gt.to(device)

        pre = model(image_noise)
        image_gt = np.array(np.transpose(image_gt[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        image_noise = np.array(np.transpose(image_noise[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        pre = np.array(np.transpose(pre[0].detach().numpy(), (1, 2, 0))*255,dtype=int)
        print(pre)
        print(" Noise : ",psnr(image_noise,image_gt), "   pre : ",psnr(pre,image_gt))
        plt.subplot(1,2,1)
        plt.imshow(image_noise)
        plt.subplot(1, 2, 2)
        plt.imshow(pre)
        plt.show()


def test_multi(noise_dir,gt_dir,image_size,num_workers,checkpoint,resume):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=1,shuffle=True, num_workers=num_workers)
    model_single = SFD_C().to(device)
    model = MFD_C(model_single).to(device)
    if resume != '':
        with open(os.path.join(checkpoint,resume), 'rb') as f:
            save_dict = torch.load(f)
            model.load_state_dict(save_dict['state_dict'])
            epoch_start = save_dict['epoch']
    for step, (image_noise, image_gt) in enumerate(data_loader):
        image_noise_batch = image_noise.to(device)
        image_gt = image_gt.to(device)
        print(image_noise_batch.size())
        batch_size_i = image_noise_batch.size()[0]
        mfinit1, mfinit2, mfinit3 = torch.zeros(3, 1, 64, image_size, image_size).to(device)
        mfinit4 = torch.zeros(1, 3, image_size, image_size).to(device)
        i = 0
        for i_burst in range(batch_size_i):
            frame = image_noise_batch[:,i_burst,:,:,:]
            print(frame.size())
            if i == 0:
                i += 1
                dframe, mf1, mf2, mf3, mf4 = model(
                    frame, mfinit1, mfinit2, mfinit3, mfinit4)
            else:
                dframe, mf1, mf2, mf3, mf4= model(frame, mf1, mf2, mf3, mf4)



if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/FullTest/noisy', help='path to noise image file')
    parser.add_argument('--gt_dir','-g',  default='/home/dell/Downloads/FullTest/clean', help='path to groud true image file')
    parser.add_argument('--image_size','-sz' , type=int,default=256, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume', '-r', type=str, default="SFD_C_0.pth.tar",
                        help='file name of checkpoint')
    parser.add_argument('--type_model', '-t', type=str, default='single',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'single':
        test_single(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.checkpoint,args.resume)
    elif args.type_model == 'multi':
        test_multi(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.checkpoint,args.resume)



