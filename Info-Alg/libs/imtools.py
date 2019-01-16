import numpy as np
from copy import deepcopy

class imres:
    """
    imres(z, mask, cutoff = 100, bits = 8)
    A class for restoring scratched image.
    
    Parameters
    ----------
    z : image tensor comes from cv2.imread with the shape (W,H,channels)
      User should aware that cv2.imread orders the color channels in BGR instead of RGB.
      This is different from matplotlib.pyplot.imshow in general.
    mask : a 2D array
      Binary image, masked positions are indicated with 1
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
        
        # Deal with binary mask image: Value greater than 0 indicates the scratch positions
        rmask,cmask = np.where(mask > 0)
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
        
        # Prior colors for the scratched pixels. Now self.z becomes the prior image
        for pos in self.mask:
            self.z[pos[0],pos[1],] = np.random.randint(0,256,self.ch)
            
        # Create an array ready for the posterior image
        self.posterior = deepcopy(self.z)
    
    def restore(self):
        """
        Initializing restoration process
        """         
        for pos in self.mask:
            for i in range(self.ch):
                # Calculate the correct color from prior image
                self.scratch(self.z[:,:,i], pos[0], pos[1])
                # Update the correct color to posterior image
                self.posterior[pos[0],pos[1],i] = self.scratch_pixel
        # After the above iteration, all the scratch pixels are corrected
        # Update the earlier prior by the posterior. This will be the prior image in the next iteration
        self.z = self.posterior
    
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
