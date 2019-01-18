import numpy as np
from copy import deepcopy

class imres:
    """
    imres(z, mask, cutoff = 1000, rad = 3, bits = 8)
    A class for restoring scratched image.
    
    Parameters
    ----------
    z : image tensor comes from cv2.imread with the shape (W,H,channels)
      User should aware that cv2.imread orders the color channels in BGR instead of RGB.
      This is different from matplotlib.pyplot.imshow in general.
    mask : a 2D array
      Binary image, masked positions are indicated with 1
    cutoff: a scalar, the truncated value for the energy function
    rad : integer
      Number of the neighbor pixels user wants to include for a given masked pixels, the
      larger the longer the computational time
    bits : integer
      Represents 2**bits color depth in each RGB channel. Sometimes it refers to 24-bit
      true color, total 1.67M colors.
      Add additional color depth will decrease the efficiency and does not improve the
      restored quality if the input image contains no color with such depth.
    """
    
    def __init__(self, z, mask, cutoff = 1000, rad = 3, bits = 8):
        """
        Define inputs and parameters
        """        
        self.r, self.c, self.ch = z.shape
        self.rad = rad
        
        # Deal with binary mask image: Value greater than 0 indicates the scratch positions
        maskpos = np.where(mask > 0)
        self.mask = (np.vstack((maskpos[0], maskpos[1])).T + np.array([rad,rad])).tolist()
        
        # How many pixels we have to correct? Uncomment this if you want to monitor the change
        # rate between iterations
        #self.pixnum = len(maskpos[0])
        
        # Padding the edges of image array with extra 0
        self.z = np.zeros((self.r+2*rad,self.c+2*rad,self.ch), dtype=np.int16)
        self.z[rad:self.r+rad,rad:self.c+rad,] = z
        
        # Generate markov blanket
        self.blanket = self.gen_blanket(rad)
        
        # Generate cutoff criteria
        self.cutoff = np.repeat([self.gen_blanket(rad,cutoff)], 2**bits, axis=0)
        
        # Create a (5,5) color level
        self.cbits = []
        for i in range(2**bits):
            x = np.full((2*rad+1,2*rad+1), i, dtype=np.int16)
            self.cbits.append(x)
        self.cbits = np.array(self.cbits)
        
        # Prior colors for the scratched pixels. Now self.z becomes the prior image
        self.z[maskpos[0]+rad,maskpos[1]+rad,] = np.random.randint(0,256,size=(len(maskpos[0]),self.ch))
    
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
        
        # Calculate the change rate between previous and this iterations, uncomment this if you want to monitor it
        #self.change = self.change_rate(self.z, posterior)
        
        # Update the prior by the posterior and can be used for the next iteration
        self.z = posterior
            
    def scratch(self,z,r,c):
        """
        Function for inferencing the color of the masked pixels, not for stand-alone use
        """
        psi = np.minimum((((self.cbits - z[r-self.rad:r+(self.rad+1),c-self.rad:c+(self.rad+1)])*self.blanket)**2), \
                         self.cutoff).sum(axis=(1,-1))
        return np.argmin(psi)
        #self.scratch_pixel = np.argmin(psi)
        
    def gen_blanket(self,radius,element=1):
        """
        Generate Markov blanket with given radius (how many neighbor pixels)
        If the element not equals 1, it is used to generate the cutoff criteria
        Not for stand-alone use
        """
        x, y = np.mgrid[0:2*radius+1,0:2*radius+1]
        blanket = ((x - radius)**2 + (y - radius)**2 <= radius**2)*element
        blanket[radius,radius] = 0
        return blanket
    
    # Uncomment the following if you want to monitor the change rate between iterations
    #def change_rate(self,prior,posterior):
    #    """
    #    Calculate the change rate between the prior and posterior images, not for stand-alone use
    #    """
    #    return np.sum(prior!=posterior)*100/(3*self.pixnum)        
        
    def status(self):
        """
        Return the current status of the restored image
        """
        return self.z[self.rad:self.r+self.rad,self.rad:self.c+self.rad,]