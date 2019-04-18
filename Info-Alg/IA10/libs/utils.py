import numpy as np


def psnr(im, tr_im):
    """
    Function for calculating PSNR index
    
    Parameter
    ---------
    im : Target image array, usually a corrupted image 
    tr_im : True, clean image
    
    Output
    ------
    Scalar, PSNR index in dB unit
    """
    mse = (np.float32(im) - np.float32(tr_im))**2
    return 10*np.log10(255**2/np.mean(mse))