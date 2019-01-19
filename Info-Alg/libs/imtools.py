import numpy as np
from copy import deepcopy

class imres:
    """
    imres(z, mask, cutoff = 1000, radi = 2, bits = 8)
    A class for restoring scratched image.
    
    Parameters
    ----------
    z : image tensor comes from cv2.imread with the shape (W,H,channels)
      User should aware that cv2.imread orders the color channels in BGR instead of RGB.
      This is different from matplotlib.pyplot.imshow in general.
    mask : a 2D array
      Binary image, masked positions are indicated with 1
    cutoff: a scalar, the truncated value for the energy function
    radi : integer
      Radius of the circular Markov blanket. Unit is in pixels
    bits : integer
      Represents 2**bits color depth in each RGB channel. Sometimes it refers to 24-bit
      true color, total 1.67M colors.
      Add additional color depth will decrease the efficiency and does not improve the
      restored quality if the input image contains no color with such depth.
      
    Return
    ------
    Restored image array with 3 color channels arranged as input
    """
    
    def __init__(self, im, mask, cutoff = 1000, radi = 2, bits = 8):
        """
        Define inputs and parameters
        """        
        self.r, self.c, self.ch = im.shape
        self.radi = radi
        self.bits = int(bits)
        
        # Deal with binary mask image: Value greater than 0 indicates the scratch positions
        maskpos = np.where(mask > 0)
        self.mask = (np.vstack((maskpos[0], maskpos[1])).T + np.array([radi,radi])).tolist()
        
        # How many pixels we have to correct? Uncomment this if you want to monitor the change rate between iterations
        self.pixnum = len(maskpos[0])
        
        # Padding the edges of image array with extra 0
        self.z = np.zeros((self.r+2*radi,self.c+2*radi,self.ch), dtype=np.int16)
        self.z[radi:self.r+radi,radi:self.c+radi,] = im
        
        # Generate markov blanket with radius = radi
        self.blanket = self.gen_blanket(radi)
        
        # Generate cutoff criteria
        self.cutoff = np.repeat([self.gen_blanket(radi,cutoff)], 2**self.bits, axis=0)
        
        # Create a (5,5) color level
        self.cbits = []
        for i in range(2**self.bits):
            x = np.full((2*radi+1,2*radi+1), i, dtype=np.int16)
            self.cbits.append(x)
        self.cbits = np.array(self.cbits)
        
        # Prior colors for the scratched pixels. Now self.z becomes the prior image
        self.z[maskpos[0]+radi,maskpos[1]+radi,] = np.random.randint(0,256,size=(len(maskpos[0]),self.ch))
    
    def restore(self):
        """
        Initializing the restoration process
        """         
        # Create an array for posterior
        posterior = deepcopy(self.z)
        
        for pos in self.mask:
            for i in range(self.ch):
                # Calculate the correct color from prior image and update the posterior
                posterior[pos[0],pos[1],i] = self.argminE(self.z[:,:,i], pos[0], pos[1])
        
        # Calculate the change rate between previous and this iterations, uncomment it if you want to monitor it
        #self.change = self.change_rate(self.z, posterior)
        
        # Update the prior by the posterior and can be used for the next iteration
        self.z = posterior
            
    def argminE(self,z,r,c):
        """
        Finding which grey value that minimizes energy, not for stand-alone use
        """
        psi = np.minimum((((self.cbits - z[r-self.radi:r+(self.radi+1),c-self.radi:c+(self.radi+1)])*self.blanket)**2), \
                         self.cutoff).sum(axis=(1,-1))
        return np.argmin(psi)
    
    def gen_blanket(self,radius, element=1): 
        """
        Generate Markov blanket with given radius (how many neighbor pixels)
        If the element not equals 1, it is used to generate the cutoff criteria
        Not for stand-alone use
        """
        x, y = np.mgrid[0:2*radius+1,0:2*radius+1]
        blanket = ((x - radius)**2 + (y - radius)**2) -(radius**2+1)
        blanket[blanket>0] = 0
        blanket[radius,radius] = 0
        return (np.abs(blanket/radius**2)*element).astype(np.float16)
    
    # Monitor the change rate between iterations
    def change_rate(self,prior,posterior):
        """
        Calculate the change rate between the prior and posterior images, not for stand-alone use
        """
        return np.sum(prior!=posterior)*100/(self.ch*self.pixnum)        
        
    def status(self):
        """
        Return the current status of the restored image
        """
        return self.z[self.radi:self.r+self.radi,self.radi:self.c+self.radi,]
