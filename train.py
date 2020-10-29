import argparse
import os
from model import *
from metric import *
from data_loader import SingleLoader,MultiLoader
from torch.utils.data import DataLoader
import torch.optim as optim

def train_single(noise_dir,gt_dir,image_size,num_workers,batch_size,n_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_every = 1
    save_every = 1
    dataset = SingleLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    model = SFD_C().to(device)
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)

    for epoch in range(n_epoch):
        for step, (image_noise, image_gt) in enumerate(data_loader):
            image_noise = image_noise.to(device)
            image_gt = image_gt.to(device)
            pre = model(image_noise)
            loss = loss_func(pre, image_gt)
            if (step + 1) % loss_every == 0:
                print('t = %d, loss = %.4f' % (step + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "SFD_C_{}_{:f}.pth.tar".format(epoch,loss))

            torch.save(save_dict, filename)

def train_multi(noise_dir,gt_dir,image_size,num_workers,batch_size,n_epoch):
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    loss_every = 1
    save_every = 1
    dataset = MultiLoader(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    model_single = SFD_C().to(device)
    model = MFD_C(model_single).to(device)
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=0.001, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)
    for epoch in range(n_epoch):
        for step, (image_noise, image_gt) in enumerate(data_loader):
            image_noise_batch = image_noise.to(device)
            image_gt = image_gt.to(device)
            print(image_noise_batch.size())
            batch_size_i = image_noise_batch.size()[0]
            mfinit1, mfinit2, mfinit3 = torch.zeros(3, batch_size, 64, image_size, image_size).to(device)
            mfinit4 = torch.zeros(batch_size, 3, image_size, image_size).to(device)
            i = 0
            for i_burst in range(batch_size_i):
                frame = image_noise_batch[:,i_burst,:,:,:]
                print(frame.size())
                if i == 0:
                    i += 1
                    dframe, mf1, mf2, mf3, mf4 = model(
                        frame, mfinit1, mfinit2, mfinit3, mfinit4)
                    loss_sfd = loss_func(dframe, image_gt)
                    loss_mfd = loss_func(mf4, image_gt)

                else:
                    dframe, mf1, mf2, mf3, mf4= model(frame, mf1, mf2, mf3, mf4)
                    loss_sfd += loss_func(dframe, image_gt)
                    loss_mfd += loss_func(mf4, image_gt)
            loss = loss_sfd + loss_mfd
            if (step + 1) % loss_every == 0:
                print('t = %d, loss = %.4f' % (step + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "SFD_C_{}_{:f}.pth.tar".format(epoch,loss))

            torch.save(save_dict, filename)
    pass


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/0001_NOISY_SRGB', help='path to noise image file')
    parser.add_argument('--gt_dir','-g',  default='/home/dell/Downloads/0001_GT_SRGB', help='path to groud true image file')
    parser.add_argument('--image_size','-sz' , type=int,default=256, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=4, type=int, help='number of workers in data loader')
    parser.add_argument('--batch_size', '-bz', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--n_epoch', '-ep', default=8, type=int, help='number of workers in data loader')
    parser.add_argument('--eval', action='store_true', help='whether to work on the evaluation mode')
    parser.add_argument('--checkpoint', '-ckpt', dest='checkpoint', type=str, default='best',
                        help='the checkpoint to eval')
    parser.add_argument('--type_model', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'single':
        train_single(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.batch_size,args.n_epoch)
    elif args.type_model == 'multi':
        train_multi(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.batch_size,args.n_epoch)



