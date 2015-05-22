import comet as c
import pyfits as p
import pyfits as p
import numpy as np
import matplotlib.pyplot as plt
import gaussian as g
from scipy.ndimage import geometric_transform

# Routine to unwrap image commented out


# take in fits image data to array 
data = p.getdata("frame.fits")

# log scale the image to make finding the comet easier.
def log_scale(data, scale_factor = 1):

        logdata = scale_factor * np.log10(1 + np.abs(data))
       
        return logdata

log_data = log_scale(data)


#p.writeto("log_scale_n655.fits", log_data)

# From viewing the previous image with square_root scale in ds9,
# I choose an empirical guess for the center of the comet's nucleus.

gcy = 1058
gcx = 977

# I will take a 7x7 sub_image around the guessed center and find the centroid.

sub = data[gcy-4:gcy+3, gcx-4:gcx+3]

centroid = c.centroid(sub)

# Using the centroid values I store a tuple of center coordinates for the
# nucleus in the entire frame.

center = (gcy - 4 + centroid[0], gcx - 4 + centroid[1])

cy = center[0]
cx = center[1]

# I now take a sub image around the centroid for the comet of size (200x200)

sub = np.copy(data[cy-100:cy+100, cx-100:cx+100])

# store new centroid values for sub image
gcx=80
gcy=80
sub_sub = np.copy(sub[gcy-3:gcy+4, gcx-3:gcx+4])
centroid = c.centroid(sub_sub)
center = (gcy - 3 + centroid[0], gcx - 3 + centroid[1])
cy = center[0]
cx = center[1]


# I will subtract off the median sky. I use a square located in the image
# where no stellar photon counts can be seen away from the comet.

med_sub = data[887:987, 1241:1341]

median = np.median(med_sub)

std = np.std(med_sub)

data -= median

sub -= median

un = c.unwrap(sub)

shape = un.shape


for i in np.arange(shape[1]):
    where = np.where(un[:,i] != 0)
    tot = 0
    for j in where:
        tot += un[j,i]
    tot = np.sum(tot)
    mean = tot / np.size(where)
    un[:,i] /= mean

p.writeto("unwrappedazmeanno.fits", un)

un = p.getdata("unwrappedazmeanno.fits")

def smooth(image, size):

    import numpy as np

    data = np.copy(image)

    # forms normalized 2D gaussian kernel array
    size = np.int(size)
    
    x, y = np.mgrid[-size:size+1, -size:size+1]
    
    g = np.exp(-(x**2/np.float(size)+y**2/np.float(size)))

    gaussian_kernel = g / np.sum(g)

    # convolves image with gaussian kernel array
    from scipy import ndimage
    
    smooth = ndimage.filters.convolve(data, gaussian_kernel)

    return smooth

def fitJet(im, yRange, xRange, xBegin):
    
    data = np.copy(im[yRange[0]:yRange[1], xRange[0]:xRange[1]])
    #plt.imshow(data, vmin=-363, vmax=1000, cmap = plt.cm.RdBu_r)
    
    y, x = data.shape

    print y, x

    xInd = np.arange(xBegin, x)
    yFits = np.asarray([])

    for i in xInd:
        if(i==xBegin):
            d = data[0:y, i]
            center = np.int(y/2)
            x = np.indices(d.shape)
            guess = (1.0, center, d[center])
            ret = g.fitgaussian(d, guess=guess, x=x)
            center = ret[1]
            center = np.round(center)
            center = np.int(center)
            yFits = np.append(yFits, ret[1])
        else:
         d = data[0:y, i]
         x = np.indices(d.shape)
         guess = (1.0, center, d[center])
         ret = g.fitgaussian(d, guess=guess, x=x)
         center = ret[1]
         center = np.round(center)
         center = np.int(center)
         yFits = np.append(yFits, ret[1])

    #plt.plot(xInd, yFits, 'r.')
    #plt.show()

    return (yFits + yRange[0]), xInd


#Change order of smoothing kernel for better fit.
unn = smooth(un,3)

plt.imshow(un, vmin=0.00017014914, vmax=1.6534909, cmap = plt.cm.bone)

# Gaussian fitting

# jet1
xRange = (0,70)
yRange = (316,347)

k = fitJet(unn, yRange, xRange, xBegin=10)

plt.plot(k[1], k[0], 'r.')

#jet2
xRange = (13,55)
yRange = (265,285)

k = fitJet(unn, yRange, xRange, xBegin=10)

plt.plot(k[1], k[0], 'r.')

#jet3        
xRange = (15,56)
yRange = (223,241)

k = fitJet(unn, yRange, xRange, xBegin=18)

plt.plot(k[1], k[0], 'r.')

plt.show()




