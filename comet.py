import numpy as np
import pyfits as p 


"""
   Functions contained in this file:

   disk()
   shift_diff()
   rot_diff()
   centroid() 
   radial_profile()
   azimuthal_mean_mask()
   azimuthal_med_mask()
   azimuhtal_mean()
   azimuthal_med()
   unsharp_masking()
   laplace()
   lar_sek()
   unwrap()
   
"""

"""
    disk(r, center, shape, less_equal = 'FALSE')

    This function creates a boolean mask of shape containing a disk or radius
    `r` of TRUE values centered at `center`.
    

    Parameters 
    ----------
    r:       Radius of disk, can be fractional.

    center:  Tuple containing center coordiates (cy, cx).

    shape:   Tuple containing size coordinates (size_y, size_x).

    less_equal: If set to !False, disk will have extra TRUE pixels in each dimension
                along dimensions from the center. When using disk in a routine like
                az_mean_mask with !FALSE, then a spike artifact appears to radiate
                from the center in each dimension. When set to FALSE, the disk is
                more square, but will not create this artifact.

    Returns 
    -------
     boolean mask of shape containing a disk or radius `r` of TRUE
     values centered at `center`.

    Examples
    --------

    >>> disk(2, (2, 2), (5, 5))
    array([[False, False,  True, False, False],
           [False,  True,  True,  True, False],
           [ True,  True,  True,  True,  True],
           [False,  True,  True,  True, False],
           [False, False,  True, False, False]], dtype=bool)

    >>> disk(3, (3, 3), (7, 7))
    array([[False, False, False,  True, False, False, False],
           [False,  True,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True,  True, False],
           [ True,  True,  True,  True,  True,  True,  True],
           [False,  True,  True,  True,  True,  True, False],
           [False,  True,  True,  True,  True,  True, False],
           [False, False, False,  True, False, False, False]], dtype=bool)
           
    References
    ----------

    Joe Harington's IDL to Python code for Ellipsoid mask.

    Revisions
    ---------
    2011-06-30  0.1  Joseph A Conenna- First Draft

  """

def disk(r, center, shape, less_equal = 'FALSE'):

  import numpy as np
  
  # indicies array
  indices      = np.indices(shape, dtype=float)

  # converts center to array containing an array of the value.
  center        = np.asarray(center)
  cshape        = np.ones(1 + center.size)
  cshape[0]     = center.size
  center.shape  = cshape

  # converts radius to an array containing an array of the value.
  r          = np.asarray(r)

  # returns boolean array of values where the sum array is less than or
  # equal to 1 using numpy power.

  if(less_equal != 'FALSE'):
      return np.sum(((indices - center)/r)**2, axis=0) <= 1.
  else:
      return np.sum(((indices - center)/r)**2, axis=0) < 1.


"""
    shift_diff(data, dy, dx)

    This function takes in a 2D data array and shifts the array by dy and dx
    and subtracts it from the data array and returns the result.

    Parameters 
    ----------
    data:    A 2D array with photon count values.

    dy:      An integer value for the y displacement.

    dx:      An integer value for the x displacement.

    Returns 
    -------
    diff: A 2D array containing the difference of the original data and the shifted
             data.

    Examples
    --------

    >>> grid = [[0, 0, 0, 0, 0],
                [0, 3, 3, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 3, 3, 3, 0],
                [0, 0, 0 ,0, 0]]

    >>> shift_diff(grid,1,1)
    array([[ 0,  0,  0,  0,  0],
           [ 0,  3,  3,  3,  0],
           [ 0,  3,  0,  0, -3],
           [ 0,  3,  0,  0, -3],
           [ 0,  0, -3, -3, -3]])
    
    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-06-30  0.1  Joseph A Conenna- First Draft

  """

def shift_diff(data, dy, dx):

    from scipy import ndimage

    shift = (dy, dx)

    shift_data = ndimage.interpolation.shift(data, shift)

    diff = data - shift_data
    
    return diff

"""
    rot_diff(data, degrees)

    This function takes in a 2D data array and rotates it CCW for positive `degrees` value
    and CW for negative `degrees` value. The new rotated array has the same shape, so there
    is data loss, but this is necessary for array subtraction. It is advisable to send in a
    subimage of a comet that is centered with ample room about the centroid to reduce data
    loss from coma. Function uses interpolation which can yield poor results on very small data
    sets when using small rotation angles. A powerful benefit of a rotational shift is that a
    rotation around the optocenter will produce small shifts near the nucleus, where the features
    tend to be small and well-defined, and increasingly larger shifts outwards where features tend
    to spread out and become diffuse.
    
    Parameters 
    ----------
    data:    A 2D array with photon count values.

    degrees: A positive or negative values in degrees to rotate the image by.

    Returns 
    -------
    rotdiff: A 2D array containing the difference of the original data and the rotated
             data.

    Examples
    --------
    >>> grid = [[0, 0, 0, 0, 0],
        	[0, 0, 3, 0, 0],
                [0, 0, 0, 0, 0],
                [0, 0, 3, 0, 0],
                [0, 0, 0 ,0, 0]]
    >>> difference = rot_diff(grid,45)
    >>> difference
    array([[ 0,  0,  0,  0,  0],
          [ 0, -2,  2,  0,  0],
          [ 0, -1,  0, -1,  0],
          [ 0,  0,  2, -2,  0],
          [ 0,  0,  0,  0,  0]])

    >>> grid = [[10, 0, 0, 0, 0],
	       [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0, 0, 0],
               [0, 0, 0 ,0, 10]]
    >>> difference = rot_diff(grid, degrees=90)
    >>> difference
    array([[ 10,   0,   0,   0, -10],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [  0,   0,   0,   0,   0],
           [-10,   0,   0,   0,  10]])

    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-06-22  0.1  Joseph A Conenna- First Draft

  """

def rot_diff(data, degrees):

    from scipy import ndimage

    rotdata = ndimage.interpolation.rotate(data, degrees, reshape = False)

    rotdiff = data - rotdata
    
    return rotdiff




"""
    centroid(data)

    This function find the centroid (center of light) of a 2D array of photon counts
    and returns the y, x coordinates.
    
    Parameters 
    ----------
    data: A 2D array with photon count values 

    Returns 
    -------
    loc: A tuple containing the y, x coordinates of the centroid.

    Examples
    --------
    >>> grid = [[0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0],
                [3, 3, 3, 0, 0],
                [3, 3, 3, 0, 0],
                [0, 0, 0 ,0, 0]]
    >>> centroid(grid)
    (2.0, 1.0)
    
    >>> grid = [[0, 0, 0, 0, 0],
                [3, 3, 3, 0, 0],
                [3, 0, 3, 0, 0],
                [3, 3, 3, 0, 0],
                [0, 0, 0 ,0, 0]]
    >>> centroid(grid)
    (2.0, 1.0)
    
    >>> grid = [[ 0,  0,  0, 0, 0],
                [-3, -3, -3, 0, 0],
                [-3, -3, -3, 0, 0],
                [-3, -3, -3, 0, 0],
                [ 0,  0,  0 ,0, 0]]
    >>> centroid(grid)
    (2.0, 1.0)
    
    >>> grid = [[ 0,  5,  1, 2, 1],
                [ 3, -8, -3, 9, 0],
                [ 3, -2,  3, 2, 7],
                [-3, -3, -3, 0, 0],
                [ 4,  0,  0, 8, 0]]
     >>> centroid(grid)
     (1.8461538461538463, 3.1923076923076925)

    References
    ----------

    S.Howell, Handbook of CCD Astronomy: 5.1.1 Image Centering, p.105

    Revisions
    ---------
    2011-06-22  0.1  Joseph A Conenna- First Draft

  """

def centroid(data):
    # initialize variables
    count_y = 0
    count_x = 0
    sum_count = 0

    # double loop, calculates centroid parameters
    for i in range(len(data)):
        for j in range(len(data[i])):
            if data[i][j] != 0:
                count_y += i * data[i][j]
                count_x += j * data[i][j]
                sum_count += data[i][j]

    # calculates and returns centroid
    loc = (float(count_y)/sum_count, float(count_x)/sum_count)

    return loc




"""
    radial_profile(data, cy, cx)

    This function creates a radial profile to be used in processing of comet
    images. The function takes in a data array containing the photon counts
    in the image. An array of equal size to the data is initialized to one.
    A double loop returns a value for each element in the radial profile array,
    the value being the distance from the centroid to the corner of each pixel
    closest to the origin. The radial profile array is returned.
    
    Parameters 
    ----------
    data:   A 2D array with photon count values.
    cy,cx:  Centroid values to be used. 

    Returns 
    -------
    rp: A 2D array containing values for each pixels distance from the centroid.

    Examples
    --------
    >>> grid = [[ 0,  0,  0,  0, 0],
                [ 0,  1,  1,  1, 0],
                [ 0,  1,  3,  1, 0],
                [ 0,  1,  1,  1, 0],
                [ 0,  0,  0,  0, 0]]
    >>> radial_profile(grid)
    array([[ 2.82842712,  2.23606798,  2.        ,  2.23606798,  2.82842712],
           [ 2.23606798,  1.41421356,  1.        ,  1.41421356,  2.23606798],
           [ 2.        ,  1.        ,  0.        ,  1.        ,  2.        ],
           [ 2.23606798,  1.41421356,  1.        ,  1.41421356,  2.23606798],
           [ 2.82842712,  2.23606798,  2.        ,  2.23606798,  2.82842712]])

    >>> grid = [[ 0,  5,  1, 2, 1],
            [ 3, -8, -3, 9, 0],
            [ 3, -2,  3, 2, 7],
            [-3, -3, -3, 0, 0],
            [ 4,  0,  0, 8, 0]]
    >>> radial_profile(grid)
    array([[ 3.68769744,  2.86609439,  2.19769917,  1.85614285,  2.01510568],
           [ 3.30254519,  2.3499339 ,  1.46204445,  0.86773186,  1.16976203],
           [ 3.19601268,  2.19769917,  1.20219228,  0.24627401,  0.82221378],
           [ 3.39443506,  2.47741276,  1.6592042 ,  1.16976203,  1.40844872],
           [ 3.85095854,  3.07331519,  2.46183892,  2.16241428,  2.3003087 ]])

    
    
    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-06-22  0.1  Joseph A Conenna- First Draft

"""

def radial_profile(data, cy, cx):

    import numpy as np

    # size of data (y, x)
    shape = np.shape(data)

    # allocates array of shape filled with ones
    rp = np.ones(shape)

    # dobule loop for every element in radial profile array
    # Calculates distance value from centroid to corner closest
    # to origin (reference point)
    for y in np.arange(shape[0]):
        for x in np.arange(shape[1]):
              rp[y][x] = np.sqrt( (y-cy)**2 + (x-cx)**2)
            
    return rp


"""
    azimuthal_mean_mask(image)

    This function uses the observed coma, gas or dust, to create a profile from the
    comet itself. By averaging azimuthally around the opto-center a radial profile
    that closely matches the shape of the real coma can be created. This algorithm
    takes the mean of values in an annulus and stores an annulus of the same size
    containing this average value. Rings are created this way from the center to
    the corners of an image. The main problem with this technique is that arc-shaped
    features or bright stars may introduce bumps into the profile, which can then produce
    artifacts in the enhanced image. If there are such features in the image, use az_avg_med()
    which uses a median rather than an average.
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    mask: A 2D array containing azimuthally averaged values around the center.

    Examples
    --------
    >>> image = np.ones(25)
    >>> image.shape = (5,5)
    >>> image = radial_profile(image)
    >>> print image
    
    [[ 2.82842712  2.23606798  2.          2.23606798  2.82842712]
     [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
     [ 2.          1.          0.          1.          2.        ]
     [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
     [ 2.82842712  2.23606798  2.          2.23606798  2.82842712]]
     
    >>> mask = az_avg_mean_mask(image)
    >>> print mask
    
    [[ 2.43352103  2.43352103  1.70710678  2.43352103  2.43352103]
     [ 2.43352103  1.70710678  1.          1.70710678  2.43352103]
     [ 1.70710678  1.          1.          1.          1.70710678]
     [ 2.43352103  1.70710678  1.          1.70710678  2.43352103]
     [ 2.43352103  2.43352103  1.70710678  2.43352103  2.43352103]]

    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def azimuthal_mean_mask(image): 
    
    # makes copy of data
    data = np.copy(image)
    
    # finds shape of data
    shape = data.shape

    # stores tuple of center coordinates (cy,cx)
    center = ( ((shape[0]-1) / 2.0), ((shape[1]-1)/2.0) )

    # finds maximum dimension to use
    max_dim = np.max(shape)

    # the algorithm needs to find the azimuthal average of every
    # pixel in the image, even those near the corners where only
    # arcs can be evaluated. The image is padded with zeros such
    # that the maximum azimuthal average taken in an annulus lies
    # on the corner pixels.
    new_dim = np.hypot(max_dim/2.0, max_dim/2.0)

    # calculates new shape of new array
    new_shape = (new_dim*2, new_dim*2)
    new_shape = np.ceil(new_shape)
    if(np.ceil(new_shape[0])%2 == 0.0):
        new_shape[0] = np.ceil(new_shape[0]) + 1
    if(np.ceil(new_shape[1])%2 == 0.0):
        new_shape[1] = np.ceil(new_shape[1]) + 1

    # calculates new center of new array
    new_center = ( np.int((new_shape[0])/2), (np.int(new_shape[1])/2) )

    # initializes new array.
    new_data = np.zeros(new_shape)

    # calculates dimensions for array slicing
    x1 = np.int(new_shape[0]) - np.int(shape[0])
    x1 /= 2
    x1 -= 1
    x2 = x1 + np.int(shape[0])
    
    y1 = np.int(new_shape[1]) - np.int(shape[1])
    y1 /= 2
    y1 -= 1
    y2 = y1 + np.int(shape[1])

    # slices new array and fits in the data inputed such that it is
    # padded with zeros.
    new_data[x1+1:x2+1, y1+1:y2+1] = data

    # initializes mask array which will be returned
    mask = np.zeros(new_shape)
    
    # for loop to loop through radius 0 to new_dim -1.
    for i in np.arange(new_dim):

        # creates a mask array with an annulus of TRUE values
        # from radius i to i+1. 
        annulus = disk(i+1, new_center, new_shape) - disk(i, new_center, new_shape)
        
        # calculates the average value within the annulus, and makes sure to
        # not included padded zeros.
        avg = new_data[np.where(annulus != 0)]
        if(np.sum(avg) == 0):
            continue
        avg = avg[np.where(avg)]
        avg = np.mean(avg)
        
        # multiplies the average value to the annulus to creates an array with
        # just a ring of this value from i to i+1.
        avgring = avg * annulus

        # adds the ring to the empty array to be returned.
        mask += avgring
        #print new_data

        #print mask
        
    # returns the proper slice of the padded array.
    return mask[x1+1:x2+1, y1+1:y2+1]


"""
    azimuthal_med_mask(image)

    This function uses the observed coma, gas or dust, to create a profile from the
    comet itself. By averaging azimuthally around the opto-center a radial profile
    that closely matches the shape of the real coma can be created. This algorithm
    takes the median of values in an annulus and stores an annulus of the same size
    containing this average value. Rings are created this way from the center to
    the corners of an image. 
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    mask: A 2D array containing azimuthally symmetric values of the median around the center.

    Examples
    --------
    >>> image = np.ones(25)
    >>> image.shape = (5,5)
    >>> image = radial_profile(image)
    >>> print image
    
    [[ 2.82842712  2.23606798  2.          2.23606798  2.82842712]
     [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
     [ 2.          1.          0.          1.          2.        ]
     [ 2.23606798  1.41421356  1.          1.41421356  2.23606798]
     [ 2.82842712  2.23606798  2.          2.23606798  2.82842712]]
     
    >>> mask = az_avg_med_mask(image)
    >>> print mask
    
    [[ 2.23606798  2.23606798  1.70710678  2.23606798  2.23606798]
     [ 2.23606798  1.70710678  1.          1.70710678  2.23606798]
     [ 1.70710678  1.          1.          1.          1.70710678]
     [ 2.23606798  1.70710678  1.          1.70710678  2.23606798]
     [ 2.23606798  2.23606798  1.70710678  2.23606798  2.23606798]]

    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def azimuthal_med_mask(image): 
    
    # makes copy of data
    data = np.copy(image)
    
    # finds shape of data
    shape = data.shape

    # stores tuple of center coordinates (cy,cx)
    center = ( ((shape[0]-1) / 2.0), ((shape[1]-1)/2.0) )

    # finds maximum dimension to use
    max_dim = np.max(shape)

    # the algorithm needs to find the azimuthal median of every
    # pixel in the image, even those near the corners where only
    # arcs can be evaluated. The image is padded with zeros such
    # that the maximum azimuthal average taken in an annulus lies
    # on the corner pixels.
    new_dim = np.hypot(max_dim/2.0, max_dim/2.0)

    # calculates new shape of new array
    new_shape = (new_dim*2, new_dim*2)
    new_shape = np.ceil(new_shape)
    if(np.ceil(new_shape[0])%2 == 0.0):
        new_shape[0] = np.ceil(new_shape[0]) + 1
    if(np.ceil(new_shape[1])%2 == 0.0):
        new_shape[1] = np.ceil(new_shape[1]) + 1

    # calculates new center of new array
    new_center = ( np.int((new_shape[0])/2), (np.int(new_shape[1])/2) )

    # initializes new array.
    new_data = np.zeros(new_shape)

    # calculates dimensions for array slicing
    x1 = np.int(new_shape[0]) - np.int(shape[0])
    x1 /= 2
    x1 -= 1
    x2 = x1 + np.int(shape[0])
    
    y1 = np.int(new_shape[1]) - np.int(shape[1])
    y1 /= 2
    y1 -= 1
    y2 = y1 + np.int(shape[1])

    # slices new array and fits in the data inputed such that it is
    # padded with zeros.
    new_data[x1+1:x2+1, y1+1:y2+1] = data

    # initializes mask array which will be returned
    mask = np.zeros(new_shape)
    
    # for loop to loop through radius 0 to new_dim -1.
    for i in np.arange(new_dim):

        # creates a mask array with an annulus of TRUE values
        # from radius i to i+1. 
        annulus = disk(i+1, new_center, new_shape) - disk(i, new_center, new_shape)
        
        # calculates the median value within the annulus, and makes sure to
        # not included padded zeros.
        med = new_data[np.where(annulus != 0)]
        if(np.sum(med) == 0):
            continue
        med = med[np.where(med)]
        med = np.median(med)
        
        # multiplies the median value to the annulus to creates an array with
        # just a ring of this value from i to i+1.
        medring = med * annulus

        # adds the ring to the empty array to be returned.
        mask += medring
        #print new_data

        #print mask
        
    # returns the proper slice of the padded array.
    return mask[x1+1:x2+1, y1+1:y2+1]


"""
    azimuthal_mean(image)

    This function calls azimuthal_mean_mask(image), divides the image by this mask, and returns
    the result.
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    result: A 2D array containing azimuthally symmetric values of the mean around the center
            divided out.

    Examples
    --------
    ...

    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def azimuthal_mean(image):

    return image / azimuthal_mean_mask(image)


"""
    azimuthal_med(image)

    This function calls azimuthal_med_mask(image), divides the image by this mask, and returns
    the result.
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    result: A 2D array containing azimuthally symmetric values of the median around the center
            divided out.

    Examples
    --------
    ...

    References
    ----------

    Comets II: Schleicher, Farnham, p.460 

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def azimuthal_med(image):

    return image / azimuthal_med_mask(image)


"""
    unsharp_masking(image, kernel)

    This function calls azimuthal_mean_mask(image), subtracts from the image this mask, and returns
    the result.
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    result: A 2D array containing original values subtracted by the gaussian smoothened values.

    Examples
    --------
    ...

    References
    ----------

    Comets II: Schleicher, Farnham, p.460

    http://en.wikipedia.org/wiki/Gaussian_blur

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def unsharp_masking(image, size):

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

    return (data - smooth) * -1


"""
    laplace(image, approx = 1)

    This function performs second order derivative edge detection by convolving the image with a
    kernel of an approximation of the Laplacian operator. 
    
    Parameters 
    ----------
    image: A 2D array with photon count values.

    Returns 
    -------
    result: A 2D array filtered. 

    Examples
    --------
    ...

    References
    ----------

    Comets II: Schleicher, Farnham, p.460

    Astronomical Image and Data Analysis: Starck, Murtagh, p.19
    - Three discrete approcximations of the Laplacian operator.

    Revisions
    ---------
    2011-07-01  0.1  Joseph A Conenna- First Draft

"""

def laplace(image, approx = 1):
    import numpy as np
    from scipy import ndimage

    data = np.copy(image)
    
    if (approx == 1):
        kernel = ([ 0, -1, 0],
                  [-1, 4, -1],
                  [ 0, -1, 0])

    if (approx == 2):
        kernel = ([-1, -1, -1],
                  [-1,  8, -1],
                  [-1, -1, -1])

    if (approx == 3):
        kernel = ([-1, -2, -1],
                  [-2,  4, -2],
                  [-1, -2, -1])
                 
    kernel = np.asarray(kernel)

    return ndimage.filters.convolve(image, kernel) * -1

"""
    lar_sek(data, degrees)

    This function uses a double subtraction method known as the Larson-Sekanina
    filter. The image is subtracted by half of the sum of the image rotated
    clockwise and counterclockwise a number of degrees.
    
    Parameters 
    ----------
    data:    A 2D array with photon count values.

    degrees: A positive value in degrees to rotate the image by.

    Returns 
    -------
    rotdiff: A 2D array containing the difference of the original data and the rotated
             data.

    Examples
    --------

    ...

    References
    ----------

    ...

    Revisions
    ---------
    2011-09-02  0.1  Joseph A Conenna- First Draft

  """

def lar_sek(data, degrees):
  
   from scipy import ndimage
   cw  =  ndimage.interpolation.rotate(data, degrees, reshape = False) 
   ccw =  ndimage.interpolation.rotate(data, -degrees, reshape = False)
   return data - 0.5*(cw + ccw)


"""
    unwrap(im)

    This function unwraps a rectangular coordinate image into a polar format
    image (r,theta).
    
    Parameters 
    ----------
    im:    A 2D array with photon count values.

    Returns 
    -------
    polar format data array
    
    Examples
    --------
    see rectToPolar.png
    ...

    References
    ----------

    ...

    Revisions
    ---------
    2011-09-02  0.1  Joseph A Conenna- First Draft

  """

from scipy.ndimage import geometric_transform
from scipy.misc import imsave

def unwrap(im):
 
 shape = np.shape(im)
 h, w = shape[0], shape[1]
 center = ( ((shape[0]-1) / 2.0), ((shape[1]-1)/2.0) )
 cy = center[0]
 cx = center[1]

 rheight = np.hypot(np.max(shape[0])/2.0, np.max(shape[1])/2.0)
  
 def rect_to_polar(xy):
     
   x,y = float(xy[1]),float(xy[0])
   
   R = x
   theta = (y / 360.0) * np.pi * 2

   # Convert x & y into pixel coordinates
   x = R * np.cos(theta) + cx
   y = R * np.sin(theta) + cy
  
   return x, y
 
 return geometric_transform(im, rect_to_polar ,output_shape=(360, rheight), order = 5, mode='constant')
