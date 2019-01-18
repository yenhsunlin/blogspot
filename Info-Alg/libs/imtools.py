import numpy as np
from copy import deepcopy

class imres:
    """
    imres(z, mask, cutoff = 1000, bits = 8)
    A class for restoring scratched image.
    
    Parameters
    ----------
    z : image tensor comes from cv2.imread with the shape (W,H,channels)
      User should aware that cv2.imread orders the color channels in BGR instead of RGB.
      This is different from matplotlib.pyplot.imshow in general.
    mask : a 2D array
      Binary image, masked positions are indicated with 1
    cutoff: a scalar, the truncated value for the energy function 
    bits : integer
      Represents 2**bits color depth in each RGB channel. Sometimes it refers to 24-bit
      true color, total 1.67M colors.
      Add additional color depth will decrease the efficiency and does not improve the
      restored quality if the input image contains no color with such depth.
    """
    
    def __init__(self, z, mask, cutoff = 1000, bits = 8):
        """
        Define inputs and parameters
        """        
        # Dimensions of the image tensor in (H,W,ch)
        self.r, self.c, self.ch = z.shape              
        
        # Deal with binary mask image: Value greater than 0 indicates the scratch positions
        maskpos = np.where(mask > 0)
        self.mask = (np.vstack((maskpos[0], maskpos[1])).T + np.array([2,2])).tolist()
        
        # How many pixels we have to correct?
        self.pixnum = len(maskpos[0])
        
        # Padding the edges of image tensor with extra 0
        # eg. (5,5) to (9,9), original image array at center
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
        
        # Create a (5,5) color level
        self.cbits = []
        for i in range(2**bits):
            x = np.full((5,5), i, dtype=np.int16)
            self.cbits.append(x)
        self.cbits = np.array(self.cbits)
        
        # Prior colors for the scratched pixels. Now self.z becomes the prior image
        self.z[maskpos[0]+2,maskpos[1]+2,] = np.random.randint(0,256,size=(len(maskpos[0]),self.ch))
    
    def restore(self):
        """
        Initializing restoration process
        """         
        # Create an array for posterior
        posterior = deepcopy(self.z)
        
        for pos in self.mask:
            for i in range(self.ch):
                # Calculate the correct color from prior image and update the posterior
                posterior[pos[0],pos[1],i] = self.scratch(self.z[:,:,i], pos[0], pos[1])
        
        # Calculate the change rate between previous and this iterations, uncomment this if you want to monitor
        #self.change = self.change_rate(self.z, posterior)
        
        # Update the prior by the posterior and can be used for the next iteration
        self.z = posterior
            
    def scratch(self,z,r,c):
        """
        Function for inferencing the masked pixels, not for stand-alone use
        """
        psi = np.minimum((((self.cbits - z[r-2:r+3,c-2:c+3])*self.blanket)**2), self.cutoff).sum(axis=(1,-1))
        return np.argmin(psi)
        #self.scratch_pixel = np.argmin(psi)
        
    #def change_rate(self,prior,posterior):
    #    """
    #    Calculate the change rate between the prior and posterior images, not for stand-alone use
    #    """
    #    return np.sum(prior!=posterior)*100/(3*self.pixnum)        
        
    def status(self):
        """
        Return the current status of the restored image
        """
        return self.z[2:self.r+2,2:self.c+2]