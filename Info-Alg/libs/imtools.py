import numpy as np

class imres:
    """
    imgres(z, mask, cutoff = 100, bits = 8)
    A class for restoring scratched image.
    
    Parameters
    ----------
    z : image tensor comes from cv2.imread with the shape (W,H,channels)
      User should aware that cv2.imread orders the color channels in BGR instead of RGB.
      This is different from matplotlib.pyplot.imshow in general.
    mask : a 2D array
      Gray scale from cv2.imread, only one channel is allowed
    cutoff: a scalar, determine the truncate value for energy function
    bits : integer
      Represents color depth, default value 8 equals 2**8 = 256 colors in each channel
    """
    
    def __init__(self, z, mask, cutoff = 100, bits = 8):
        """
        Define inputs and parameters
        """        
        # Dimensions of the image tensor in (H,W,ch)
        self.r, self.c, self.ch = z.shape              
        
        # Deal with mask image: get the positions of the masked pixels
        rmask,cmask = np.where(mask >= 250)
        self.mask = (np.vstack((rmask, cmask)).T + np.array([2,2])).tolist()
        
        # Padding the edges of image tensor with extra 0
        self.z = np.zeros((self.r+4,self.c+4,self.ch), dtype=np.int16)
        self.z[2:self.r+2,2:self.c+2,] = z
        
        self.cutoff = np.repeat([[[  0   ,cutoff,cutoff,cutoff,  0   ],   \
                                  [cutoff,cutoff,cutoff,cutoff,cutoff],   \
                                  [cutoff,cutoff,  0   ,cutoff,cutoff],   \
                                  [cutoff,cutoff,cutoff,cutoff,cutoff],   \
                                  [  0   ,cutoff,cutoff,cutoff,  0   ]]], 2**bits, axis=0)
        
        # The shape of Markov blanket
        self.blanket = np.array([[0,1,1,1,0], \
                                 [1,1,1,1,1], \
                                 [1,1,0,1,1], \
                                 [1,1,1,1,1], \
                                 [0,1,1,1,0]])
        
        # Create a (3,3) color level
        self.cbits = []
        for i in range(2**bits):
            x = np.full((5,5), i, dtype=np.int16)
            self.cbits.append(x)
        self.cbits = np.array(self.cbits)
        
        # Prior colors for the masked pixels
        for pos in self.mask:
            self.z[pos[0],pos[1],] = np.random.randint(0,256,self.ch)
    
    def restore(self):
        """
        Initializing restoration process
        """         
        for pos in self.mask:
            for i in range(self.ch):
                self.scratch(self.z[:,:,i], pos[0], pos[1])
                self.z[pos[0],pos[1],i] = self.scratch_pixel
    
    def scratch(self,z,r,c):
        """
        Function for inferencing the masked pixels, not for stand-alone use
        """
        psi = np.minimum((((self.cbits - z[r-2:r+3,c-2:c+3])*self.blanket)**2), self.cutoff).sum(axis=(1,-1))
        self.scratch_pixel = np.argmin(psi)
        
    def status(self):
        """
        Return the current status of the restored image
        """
        return self.z[2:self.r+2,2:self.c+2]
