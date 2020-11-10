import argparse
import os
from model_DGF import *
from metric import *
from data_loader_DGF import SingleLoader_DGF,MultiLoader_DGF
from torch.utils.data import DataLoader
import torch.optim as optim

def train_single(noise_dir,gt_dir,image_size,num_workers,batch_size,n_epoch,checkpoint,resume_single,loss_every,save_every,learning_rate):
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = SingleLoader_DGF(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    model = SFD_C_DGF().to(device)
    epoch_start = 0
    if resume_single != '':
        save_dict = torch.load(os.path.join(checkpoint,resume_single))
        model.load_state_dict(save_dict['state_dict'])
        epoch_start = save_dict['epoch']
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)

    for epoch in range(epoch_start,n_epoch):
        for step, (image_noise_hr,image_noise_lr, image_gt_hr) in enumerate(data_loader):
            # print("image_noise_hr  : ",image_noise_hr.size())
            # print("image_noise_lr  : ",image_noise_lr.size())
            image_noise_hr = image_noise_hr.to(device)
            image_noise_lr = image_noise_lr.to(device)
            image_gt_hr = image_gt_hr.to(device)
            pre = model(image_noise_lr,image_noise_hr)

            loss = loss_func(pre, image_gt_hr)
            if (step + 1) % loss_every == 0:
                print('single t = %d, loss = %.4f' % (step + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "SFD_C_{}.pth.tar".format(epoch))

            torch.save(save_dict, filename)
            # torch.save(model.state_dict(), filename)

def train_multi(noise_dir,gt_dir,image_size,image_size_lr,num_workers,batch_size,n_epoch,checkpoint,resume_single,resume_multi,loss_every,save_every,learning_rate):
    if not os.path.isdir(checkpoint):
        os.mkdir(checkpoint)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    dataset = MultiLoader_DGF(noise_dir=noise_dir,gt_dir=gt_dir,image_size=image_size,image_size_lr=image_size_lr)
    data_loader = DataLoader(dataset, batch_size=batch_size,shuffle=True, num_workers=num_workers)
    model_single = SFD_C_DGF().to(device)
    epoch_start=0
    if resume_multi != '':
        model = MFD_C_DFG(model_single).to(device)
        save_dict = torch.load(os.path.join(checkpoint,resume_multi))
        model.load_state_dict(save_dict['state_dict'])
        epoch_start = save_dict['epoch']
    elif resume_single != "":
        save_dict = torch.load(os.path.join(checkpoint, resume_single))
        model_single.load_state_dict(save_dict['state_dict'])
        model = MFD_C_DFG(model_single).to(device)
    else:
        model = MFD_C_DFG(model_single).to(device)
    loss_func = nn.L1Loss().cuda()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate, betas=(0.9, 0.999), eps=1e-08, weight_decay=0.00001,
                           amsgrad=False)
    model.train()
    for epoch in range(epoch_start,n_epoch):
        for step, (image_noise_hr,image_noise_lr, image_gt_hr) in enumerate(data_loader):
            image_noise_hr_batch = image_noise_hr.to(device)
            image_noise_lr_batch = image_noise_lr.to(device)
            image_gt_hr = image_gt_hr.to(device)
            # print("image_noise_hr_batch  : ",image_noise_hr_batch.size())
            # print("image_noise_lr_batch  : ",image_noise_lr_batch.size())
            # print("image_gt_hr   : ",image_gt_hr.size())
            # print(image_noise_batch.size())
            batch_size_i = image_noise_hr_batch.size()[0]
            burst_size = image_noise_hr_batch.size()[1]
            mfinit1, mfinit2, mfinit3,mfinit4,mfinit5,mfinit6,mfinit7 = torch.zeros(7, batch_size_i, 64, image_size_lr, image_size_lr).to(device)
            mfinit8 = torch.zeros(batch_size_i, 3, image_size_lr, image_size_lr).to(device)
            i = 0
            for i_burst in range(burst_size):
                frame_hr = image_noise_hr_batch[:,i_burst,:,:,:]
                frame_lr = image_noise_lr_batch[:,i_burst,:,:,:]
                # print("frame size  ",frame.size())
                if i == 0:
                    i += 1
                    dframe_lr,dframe_hr, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8,mf8_hr = model(
                        frame_lr,frame_hr, mfinit1, mfinit2, mfinit3, mfinit4,mfinit5,mfinit6,mfinit7,mfinit8)
                    loss_sfd = loss_func(dframe_hr, image_gt_hr)
                    loss_mfd = loss_func(mf8_hr, image_gt_hr)

                else:
                    dframe_lr,dframe_hr, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8 , mf8_hr= model(frame_lr,frame_hr, mf1, mf2, mf3, mf4,mf5, mf6, mf7, mf8)
                    loss_sfd += loss_func(dframe_hr, image_gt_hr)
                    loss_mfd += loss_func(mf8_hr, image_gt_hr)
            loss = loss_sfd + loss_mfd
            if (step + 1) % loss_every == 0:
                print('epoch %d  multi t = %d, loss = %.4f' % (epoch,step + 1, loss.data))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        if (epoch + 1) % save_every == 0:
            save_dict = {
                'epoch': epoch,
                'state_dict': model.state_dict(),
                'optimizer': optimizer.state_dict(),
            }
            filename = os.path.join("checkpoint", "MFD_C_{}.pth.tar".format(epoch))

            torch.save(save_dict, filename)
    pass


if __name__ == "__main__":
    # argparse
    parser = argparse.ArgumentParser(description='parameters for training')
    parser.add_argument('--noise_dir','-n', default='/home/dell/Downloads/noise', help='path to noise image file')
    parser.add_argument('--gt_dir','-g',  default='/home/dell/Downloads/gt', help='path to groud true image file')
    parser.add_argument('--image_size','-sz' , type=int,default=256, help='size of image')
    parser.add_argument('--image_size_lr','-szl' , type=int,default=64, help='size of image')
    parser.add_argument('--num_workers', '-nw', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--batch_size', '-bs', default=2, type=int, help='number of workers in data loader')
    parser.add_argument('--n_epoch', '-ep', default=5, type=int, help='number of workers in data loader')
    parser.add_argument('--learning_rate', '-lr', default=0.001, type=float, help='number of workers in data loader')
    parser.add_argument('--loss_every', '-le', default=1, type=int, help='number of inter to print loss')
    parser.add_argument('--save_every', '-se', default=5, type=int, help='number of epoch to save checkpoint')
    parser.add_argument('--checkpoint', '-ckpt', type=str, default='checkpoint',
                        help='the folder checkpoint to save')
    parser.add_argument('--resume_multi', '-rm', type=str, default='',
                        help='file name of checkpoint')
    parser.add_argument('--resume_single', '-rs', type=str, default='',
                        help='file name of checkpoint')
    parser.add_argument('--type_model', '-t', type=str, default='multi',help='type model train is single or multi')
    args = parser.parse_args()
    #
    if args.type_model == 'single':
        train_single(args.noise_dir,args.gt_dir,args.image_size,args.num_workers,args.batch_size,args.n_epoch,args.checkpoint,args.resume_single,args.loss_every,args.save_every,args.learning_rate)
    elif args.type_model == 'multi':
        train_multi(args.noise_dir,args.gt_dir,args.image_size,args.image_size_lr,args.num_workers,args.batch_size,args.n_epoch,args.checkpoint,args.resume_single,args.resume_multi,args.loss_every,args.save_every,args.learning_rate)



