
import numpy as np


def psnr(im1, im2, check=False):
    # im2clip=np.clip(im2,0,255.0)
    im2clip = im2 * 1.0
    # print(im2)
    eccart = np.mean((im1 - im2clip) ** 2)
    # maxvalue=np.max(im1)
    # maxvalue=1.0
    maxvalue = 255.0
    # print("eccart   111 : ",eccart)
    if check:
        print('Valeur moyenne image= %f' % np.mean(im2))
        print('Valeur max image= %f' % np.max(im2))
        print('Valeur min image= %f' % np.min(im2))

    PSNR = 10 * np.log10(maxvalue ** 2 / eccart)
    # print('PSNR= %f' %PSNR)

    return (PSNR)