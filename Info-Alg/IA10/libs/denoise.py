import numpy as np
from copy import copy


class denoise_det:

    def __init__(self, im, lam = 1, cutoff = 1000):
        """
        Class for denoising noisy image. This one uses determistic algorithm.
        
        Parameter
        ---------
        im : Noisy image
        lam : lambda for penalty
        cutoff : truncated value
        """
        # Determing the input is color or grey
        if len(im.shape) == 3:
            self.im = np.pad(im, ((1,1),(1,1),(0,0)), mode = 'constant')
            self.row, self.col, self.ch = self.im.shape
            self.color = True
        elif len(im.shape) == 2:
            self.im = np.pad(im, 1, mode = 'constant')
            self.row, self.col = self.im.shape
            self.color = False
        else: 
            raise ValueError('Input is not an image!')
        
        self.lam = lam
        self.cutoff = cutoff
        self.prior = np.int32(self.im.copy())
        self.denoise_im = np.int32(np.zeros_like(self.im))

        
    def denoise(self):
        """
        Initializing denoise process
        """
        if self.color:
            for ch in range(self.ch):
                for r in range(1,self.row-1):
                    for c in range(1,self.col-1):
                        self.denoise_im[r,c,ch] = np.argmin(energy(self.im[r,c,ch],self.prior[r-1:r+2,c-1:c+2,ch],self.lam,self.cutoff))
            self.prior = self.denoise_im
            
        else:
            for r in range(1,self.row-1):
                for c in range(1,self.col-1):
                    self.denoise_im[r,c] = np.argmin(energy(self.im[r,c],self.prior[r-1:r+2,c-1:c+2],self.lam,self.cutoff))
            self.prior = self.denoise_im
    
    def status(self):
        """
        Return current status of denoised image
        """
        return np.uint8(self.denoise_im[1:self.row-1,1:self.col-1])


class denoise_prob:

    def __init__(self, im, lam = 1, cutoff = 1000):
        """
        Class for denoising noisy image. This one uses probabilistic algorithm.
        
        Parameter
        ---------
        im : Noisy image
        lam : lambda for penalty
        cutoff : truncated value
        """
        # Determing the input is color or grey
        if len(im.shape) == 3:            
            self.im = np.pad(im, ((1,1),(1,1),(0,0)), mode = 'constant')
            self.row, self.col, self.ch = self.im.shape
            self.color = True
        elif len(im.shape) == 2:
            self.im = np.pad(im, 1, mode = 'constant')
            self.row, self.col = self.im.shape
            self.color = False
        else: 
            raise ValueError('Input is not an image!')
        
        self.lam = lam
        self.cutoff = cutoff
        self.prior = np.int32(self.im.copy())
        self.denoise_im = np.int32(np.zeros_like(self.im))

        
    def denoise(self):
        """
        Initializing denoise process
        """
        if self.color:
            for ch in range(self.ch):
                for r in range(1,self.row-1):
                    for c in range(1,self.col-1):
                        ene = energy(self.im[r,c,ch],self.prior[r-1:r+2,c-1:c+2,ch],self.lam,self.cutoff)
                        self.denoise_im[r,c,ch] = pix_sampling(ene)
            self.prior = self.denoise_im
            
        else:
            for r in range(1,self.row-1):
                for c in range(1,self.col-1):
                    ene = energy(self.im[r,c],self.prior[r-1:r+2,c-1:c+2],self.lam,self.cutoff)
                    self.denoise_im[r,c] = pix_sampling(ene)
            self.prior = self.denoise_im
    
    def status(self):
        """
        Return current status of denoised image
        """
        return np.uint8(self.denoise_im[1:self.row-1,1:self.col-1])


def energy(center, nei, lam, cutoff, pixels = np.arange(256)):
    """
    Calculating the energies of 256 pixel values for the center pixel in the box, Eq. (7)
    """
    nei = np.asarray(nei)
    Phi = (pixels-center)**2
    Psi = lam*(np.clip((pixels-nei[0,1])**2,0,cutoff) +   \
                np.clip((pixels-nei[1,0])**2,0,cutoff) +  \
                np.clip((pixels-nei[1,2])**2,0,cutoff) +  \
                np.clip((pixels-nei[2,1])**2,0,cutoff))
    return (Psi+Phi)


def pix_sampling(ene):
    """
    Sampling possible pixel value by given energies for all 256 values, Eq. (10)
    
    Parameter
    ---------
    ene : Energy array for 256 pixel values, the first value corresponds
          topixel value 0, and the second corresponds to pixel value 1...
          and so on
          
    Output
    ------
    Scalar, the possible pixel value
    """
    ener = np.asarray(ene-np.min(ene))
    numerator = np.exp(-ener)
    denominator = np.sum(numerator)
    prob = numerator/denominator # Eq. (10)
    return np.argmax(np.random.multinomial(1,prob))


def denoise_multif(imls):
    """
    Denoising function from multi-noisy frames
    
    Parameter
    ---------
    imls : Image list contains multple noisy frames. Each element
           is an image array
           
    Output
    ------
    A denoised image array
    
    """
    im = np.float32(imls)
    clean = np.sum(im,axis=0)/len(im)
    return np.uint8(clean)
