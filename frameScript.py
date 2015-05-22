import comet as c
import pyfits as p
import pyfits as p
import numpy as np


""" example script to process CCD data using image processing routines contained in
    comet.py module. 
"""

# take in fits image data to array 
data = p.getdata("frame.fits")

# log scale the image to make finding the comet easier.
def log_scale(data, scale_factor = 1):

        logdata = scale_factor * np.log10(1 + np.abs(data))
       
        return logdata

log_data = log_scale(data)


# Initial write of log scaled image
#p.writeto("log_scale_frame.fits", log_data)

# From viewing the previous image with square_root scale in ds9,
# I choose a guess for the center of the comet's nucleus. Centroid
# guess values are empirical values that will become concrete in centroid
# detection.

gcy = 880
gcx = 1029

# Take a 7x7 sub_image around the guessed center and find the centroid.

sub = data[gcy-4:gcy+3, gcx-4:gcx+3]

centroid = c.centroid(sub)

# Using the centroid values I store a tuple of center coordinates for the
# nucleus in the entire frame.

center = (gcy - 4 + centroid[0], gcx - 4 + centroid[1])

cy = center[0]
cx = center[1]

# I now take a sub image around the centroid for the comet of size (160x160)
# 80!

sub = np.copy(data[cy-80:cy+80, cx-80:cx+80])

# store new centroid values for sub image
gcx=80
gcy=80
sub_sub = np.copy(sub[gcy-3:gcy+4, gcx-3:gcx+4])
centroid = c.centroid(sub_sub)
center = (gcy - 3 + centroid[0], gcx - 3 + centroid[1])
cy = center[0]
cx = center[1]


# I will subtract off the median sky. I use a square empirically located in
# the image where no stellar photon counts can be seen away from the comet.

med_sub = data[1200:1300, 1200:1300]

median = np.median(med_sub)

std = np.std(med_sub)

data -= median

sub -= median


# save sub image
#p.writeto("frame_sub.fits", sub)

# First I will display a Linear Shift Difference of 5 pixels in the y & x

linear_shift_5_5 = c.shift_diff(sub, 5, 5)

#p.writeto("sub_linear_shift_diff_5_5.fits", linear_shift_5_5)


# Now I will show a Rotational Shift Difference of 3, 10, 20, 30, 40, 50 degrees.

rot_shift_3 = c.rot_diff(sub, 3)

#p.writeto("sub_rotational_diff_3_deg.fits", rot_shift_3)

rot_shift_10 = c.rot_diff(sub, 10)

#p.writeto("sub_rotational_diff_10_deg.fits", rot_shift_10)

rot_shift_20 = c.rot_diff(sub, 20)

#p.writeto("sub_rotational_diff_20_deg.fits", rot_shift_20)

rot_shift_30 = c.rot_diff(sub, 30)

#p.writeto("sub_rotational_diff_30_deg.fits", rot_shift_20)

rot_shift_40 = c.rot_diff(sub, 40)

#p.writeto("sub_rotational_diff_40_deg.fits", rot_shift_40)

rot_shift_50 = c.rot_diff(sub, 50)

#p.writeto("sub_rotational_diff_50_deg.fits", rot_shift_50)


# Now I will divide out a 1/rho profile.

radial_profile = np.copy(sub) * c.radial_profile(sub, cy, cx)

#p.writeto("sub_radial_profile_divided.fits", radial_profile)

# Set to square_root scale and see a difference in the coma.


# I will now divide out an Azimuthally Averaged profile of the bulk coma.

az_mean = c.azimuthal_mean(sub)

#p.writeto("sub_azimuthal_mean_div.fits", az_mean)


# Now I will divide out an  Azimuthal Median profile

az_med = c.azimuthal_med(sub)

#p.writeto("sub_azimuthal_med_div.fits", az_med)

# Now I will use Unsharp Masking to convolve a 3-pixel Gaussian Kernel with
# the sub and subtract it from the unaltered sub.

unsharp = c.unsharp_masking(sub, 3)

#p.writeto("sub_unsharp_masking.fits", unsharp)


# Now I will convolve an approximation of the Laplacian operator to form
# a second order derivative edge detecting Laplace Filter.

lap = c.laplace(sub, approx=1)

#p.writeto("sub_laplace_filter.fits", lap)


# Time for Larson Sekanina Filter, I will use a larger sub frame

gcy = 880
gcx = 1029

# I will take a 7x7 sub_image around the guessed center and find the centroid.


#############
center_sub = data[gcy-4:gcy+3, gcx-4:gcx+3]

centroid = c.centroid(center_sub)

# Using the centroid values I store a tuple of center coordinates for the
# nucleus in the entire frame.

center = (gcy - 4 + centroid[0], gcx - 4 + centroid[1])

cy = center[0]
cx = center[1]

##############

lrssub = np.copy(data[cy-200:cy+200, cx-200:cx+200])

# will do 1, 3, 5, 7, 10, 15, 20, 25, 30 degree rotations

lrs = c.lar_sek(lrssub, 1)
#p.writeto("LRS_Filter_1.fits", lrs)
lrs = c.lar_sek(lrssub, 3)
#p.writeto("LRS_Filter_3.fits", lrs)
lrs = c.lar_sek(lrssub, 6)
#p.writeto("LRS_Filter_6.fits", lrs)
lrs = c.lar_sek(lrssub, 7)
#p.writeto("LRS_Filter_7.fits", lrs)
lrs = c.lar_sek(lrssub, 10)
#p.writeto("LRS_Filter_10.fits", lrs)
lrs = c.lar_sek(lrssub, 15)
#p.writeto("LRS_Filter_15.fits", lrs)
lrs = c.lar_sek(lrssub, 20)
#p.writeto("LRS_Filter_20.fits", lrs)
lrs = c.lar_sek(lrssub, 25)
#p.writeto("LRS_Filter_25.fits", lrs)
lrs = c.lar_sek(lrssub, 30)
#p.writeto("LRS_Filter_30.fits", lrs)


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

    # returns original image subtracted by smoothened image

    return smooth

smooth_3 = smooth(sub,3)
lap_3_smooth = ndimage.filters.convolve(smooth_3, kernel)
#p.writeto("lap_7_smooth_3.fits",lap_9_smooth)

smooth_3 = smooth(sub,15)
lap_9_smooth = ndimage.filters.convolve(smooth_3, kernel)
#p.writeto("lap_7_smooth_15.fits",lap_9_smooth)

us = c.unsharp_masking(sub,45)
#p.writeto("us_45.fits", us)


