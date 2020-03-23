
# coding: utf-8

# In[1]:


import SimpleITK as sitk
import numpy as np
import os
#import gui


# In[2]:


INPUT_DIR ='/work/MICCAI2019/DeepLearningModels/QC_Combined/VAEGAN_OCNN_Pred_V2/tmp/'

OUTPUT_DIR='/work/MICCAI2019/DeepLearningModels/QC_Combined/VAEGAN_OCNN_Pred_V2/Q4_Augmented/'

data = []

for filename in os.listdir(INPUT_DIR):
    if filename.endswith(".nrrd"): 
        image = sitk.ReadImage(os.path.join(INPUT_DIR, filename))
        print("Pixel Type    {}".format(image.GetPixelID()))
        print("Size          {}".format(image.GetSize()))
        print("Origin        {}".format(image.GetOrigin()))
        print("Spacing       {}".format(image.GetSpacing()))
        print("Direction     {}".format(image.GetDirection()))
        data.append(image)

# Comment out the following line if you want to work in 3D. Note that in 3D some of the notebook visualizations are 
# disabled. 
#data = [data[0][:,160,:], data[1][:,160,:], data[2][:,160,:]]


# In[3]:


#def disp_images(images, fig_size, wl_list=None):
#    if images[0].GetDimension()==2:
#      gui.multi_image_display2D(image_list=images, figure_size=fig_size, window_level_list=wl_list)
#    else:
#      gui.MultiImageDisplay(image_list=images, figure_size=fig_size, window_level_list=wl_list)
    
#disp_images(data, fig_size=(6,2))


# In[5]:


def parameter_space_regular_grid_sampling(*transformation_parameters):
    '''
    Create a list representing a regular sampling of the parameter space.     
    Args:
        *transformation_paramters : two or more numpy ndarrays representing parameter values. The order 
                                    of the arrays should match the ordering of the SimpleITK transformation 
                                    parameterization (e.g. Similarity2DTransform: scaling, rotation, tx, ty)
    Return:
        List of lists representing the regular grid sampling.
        
    Examples:
        #parameterization for 2D translation transform (tx,ty): [[1.0,1.0], [1.5,1.0], [2.0,1.0]]
        >>>> parameter_space_regular_grid_sampling(np.linspace(1.0,2.0,3), np.linspace(1.0,1.0,1))        
    '''
    return [[np.asscalar(p) for p in parameter_values] 
            for parameter_values in np.nditer(np.meshgrid(*transformation_parameters))]

def similarity3D_parameter_space_regular_sampling(thetaX, thetaY, thetaZ, tx, ty, tz, scale):
    '''
    Create a list representing a regular sampling of the 3D similarity transformation parameter space. As the
    SimpleITK rotation parameterization uses the vector portion of a versor we don't have an 
    intuitive way of specifying rotations. We therefor use the ZYX Euler angle parametrization and convert to
    versor.
    Args:
        thetaX, thetaY, thetaZ: numpy ndarrays with the Euler angle values to use.
        tx, ty, tz: numpy ndarrays with the translation values to use.
        scale: numpy array with the scale values to use.
    Return:
        List of lists representing the parameter space sampling (vx,vy,vz,tx,ty,tz,s).
    '''
    return [list(eul2quat(parameter_values[0],parameter_values[1], parameter_values[2])) + 
            [np.asscalar(p) for p in parameter_values[3:]] for parameter_values in np.nditer(np.meshgrid(thetaX, thetaY, thetaZ, tx, ty, tz, scale))]
    

def eul2quat(ax, ay, az, atol=1e-8):
    '''
    Translate between Euler angle (ZYX) order and quaternion representation of a rotation.
    Args:
        ax: X rotation angle in radians.
        ay: Y rotation angle in radians.
        az: Z rotation angle in radians.
        atol: tolerance used for stable quaternion computation (qs==0 within this tolerance).
    Return:
        Numpy array with three entries representing the vectorial component of the quaternion.

    '''
    # Create rotation matrix using ZYX Euler angles and then compute quaternion using entries.
    cx = np.cos(ax)
    cy = np.cos(ay)
    cz = np.cos(az)
    sx = np.sin(ax)
    sy = np.sin(ay)
    sz = np.sin(az)
    r=np.zeros((3,3))
    r[0,0] = cz*cy 
    r[0,1] = cz*sy*sx - sz*cx
    r[0,2] = cz*sy*cx+sz*sx     

    r[1,0] = sz*cy 
    r[1,1] = sz*sy*sx + cz*cx 
    r[1,2] = sz*sy*cx - cz*sx

    r[2,0] = -sy   
    r[2,1] = cy*sx             
    r[2,2] = cy*cx

    # Compute quaternion: 
    qs = 0.5*np.sqrt(r[0,0] + r[1,1] + r[2,2] + 1)
    qv = np.zeros(3)
    # If the scalar component of the quaternion is close to zero, we
    # compute the vector part using a numerically stable approach
    if np.isclose(qs,0.0,atol): 
        i= np.argmax([r[0,0], r[1,1], r[2,2]])
        j = (i+1)%3
        k = (j+1)%3
        w = np.sqrt(r[i,i] - r[j,j] - r[k,k] + 1)
        qv[i] = 0.5*w
        qv[j] = (r[i,j] + r[j,i])/(2*w)
        qv[k] = (r[i,k] + r[k,i])/(2*w)
    else:
        denom = 4*qs
        qv[0] = (r[2,1] - r[1,2])/denom;
        qv[1] = (r[0,2] - r[2,0])/denom;
        qv[2] = (r[1,0] - r[0,1])/denom;
    return qv


# In[6]:


dimension = data[0].GetDimension()

# Physical image size corresponds to the largest physical size in the training set, or any other arbitrary size.
reference_physical_size = np.zeros(dimension)
for img in data:
    reference_physical_size[:] = [(sz-1)*spc if sz*spc>mx  else mx for sz,spc,mx in zip(img.GetSize(), img.GetSpacing(), reference_physical_size)]

# Create the reference image with a zero origin, identity direction cosine matrix and dimension     
reference_origin = np.zeros(dimension)
reference_direction = np.identity(dimension).flatten()

# Select arbitrary number of pixels per dimension, smallest size that yields desired results 
# or the required size of a pretrained network (e.g. VGG-16 224x224), transfer learning. This will 
# often result in non-isotropic pixel spacing.
reference_size = [128,128,128]
print(reference_size)
reference_spacing = [ phys_sz/(sz-1) for sz,phys_sz in zip(reference_size, reference_physical_size) ]

# Another possibility is that you want isotropic pixels, then you can specify the image size for one of
# the axes and the others are determined by this choice. Below we choose to set the x axis to 128 and the
# spacing set accordingly. 
# Uncomment the following lines to use this strategy.
#reference_size_x = 128
#reference_spacing = [reference_physical_size[0]/(reference_size_x-1)]*dimension
#reference_size = [int(phys_sz/(spc) + 1) for phys_sz,spc in zip(reference_physical_size, reference_spacing)]

reference_image = sitk.Image(reference_size, data[0].GetPixelIDValue())
reference_image.SetOrigin(reference_origin)
reference_image.SetSpacing(reference_spacing)
reference_image.SetDirection(reference_direction)

# Always use the TransformContinuousIndexToPhysicalPoint to compute an indexed point's physical coordinates as 
# this takes into account size, spacing and direction cosines. For the vast majority of images the direction 
# cosines are the identity matrix, but when this isn't the case simply multiplying the central index by the 
# spacing will not yield the correct coordinates resulting in a long debugging session. 
reference_center = np.array(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))


# In[7]:


def augment_images_spatial(original_image, reference_image, T0, T_aug, transformation_parameters,
                    output_prefix, output_suffix,
                    interpolator = sitk.sitkNearestNeighbor, default_intensity_value = 0.0):
    '''
    Generate the resampled images based on the given transformations.
    Args:
        original_image (SimpleITK image): The image which we will resample and transform.
        reference_image (SimpleITK image): The image onto which we will resample.
        T0 (SimpleITK transform): Transformation which maps points from the reference image coordinate system 
            to the original_image coordinate system.
        T_aug (SimpleITK transform): Map points from the reference_image coordinate system back onto itself using the
               given transformation_parameters. The reason we use this transformation as a parameter
               is to allow the user to set its center of rotation to something other than zero.
        transformation_parameters (List of lists): parameter values which we use T_aug.SetParameters().
        output_prefix (string): output file name prefix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        output_suffix (string): output file name suffix (file name: output_prefix_p1_p2_..pn_.output_suffix).
        interpolator: One of the SimpleITK interpolators.
        default_intensity_value: The value to return if a point is mapped outside the original_image domain.
    '''
    all_images = [] # Used only for display purposes in this notebook.
    for current_parameters in transformation_parameters:
        T_aug.SetParameters(current_parameters)        
        # Augmentation is done in the reference image space, so we first map the points from the reference image space
        # back onto itself T_aug (e.g. rotate the reference image) and then we map to the original image space T0.
        T_all = sitk.Transform(T0)
        T_all.AddTransform(T_aug)
        aug_image = sitk.Resample(original_image, reference_image, T_all,
                                  interpolator, default_intensity_value)
        sitk.WriteImage(aug_image, output_prefix + '_' + 
                        '_'.join(str(param) for param in current_parameters) +'_.' + output_suffix)
         
        all_images.append(aug_image) # Used only for display purposes in this notebook.
    return all_images # Used only for display purposes in this notebook.


# In[8]:


aug_transform = sitk.Similarity2DTransform() if dimension==2 else sitk.Similarity3DTransform()

all_images = []

for index,img in enumerate(data):
    # Transform which maps from the reference_image to the current img with the translation mapping the image
    # origins to each other.
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    # Modify the transformation to align the centers of the original and reference image instead of their origins.
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)

    # Set the augmenting transform's center so that rotation is around the image center.
    aug_transform.SetCenter(reference_center)
    
    if dimension == 2:
        # The parameters are scale (+-10%), rotation angle (+-10 degrees), x translation, y translation
        transformation_parameters_list = parameter_space_regular_grid_sampling(np.linspace(0.9,1.1,2),
                                                                               np.linspace(-np.pi/18.0,np.pi/18.0,2),
                                                                               np.linspace(-10,10,2),
                                                                               np.linspace(-10,10,2))
    else:    
        transformation_parameters_list = similarity3D_parameter_space_regular_sampling(np.random.uniform(-np.pi/18.0,np.pi/18.0,1),
                                                                                       np.random.uniform(-np.pi/18.0,np.pi/18.0,1),
                                                                                       np.random.uniform(-np.pi/18.0,np.pi/18.0,1),
                                                                                       np.random.uniform(-10,10,1),
                                                                                       np.random.uniform(-10,10,1),
                                                                                       np.random.uniform(-10,10,1),
                                                                                       np.random.uniform(0.9,1.1,1))
    generated_images = augment_images_spatial(img, reference_image, centered_transform, 
                                       aug_transform, transformation_parameters_list, 
                                       os.path.join(OUTPUT_DIR, 'spatial_aug'+str(index)), 'nrrd')
    
    #if dimension==2: # in 2D we join all of the images into a 3D volume which we use for display.
    #    all_images.append(sitk.JoinSeries(generated_images))
# If working in 2D, display the resulting set of images.    
#if dimension==2:
#    gui.MultiImageDisplay(image_list=all_images, shared_slider=True, figure_size=(6,2))


# In[9]:

'''
# using an affine transformation

flipped_images = []
for index,img in enumerate(data):
    # Compute the transformation which maps between the reference and current image (same as done above).
    transform = sitk.AffineTransform(dimension)
    transform.SetMatrix(img.GetDirection())
    transform.SetTranslation(np.array(img.GetOrigin()) - reference_origin)
    centering_transform = sitk.TranslationTransform(dimension)
    img_center = np.array(img.TransformContinuousIndexToPhysicalPoint(np.array(img.GetSize())/2.0))
    centering_transform.SetOffset(np.array(transform.GetInverse().TransformPoint(img_center) - reference_center))
    centered_transform = sitk.Transform(transform)
    centered_transform.AddTransform(centering_transform)
    
    flipped_transform = sitk.AffineTransform(dimension)    
    flipped_transform.SetCenter(reference_image.TransformContinuousIndexToPhysicalPoint(np.array(reference_image.GetSize())/2.0))
    if dimension==2: # matrices in SimpleITK specified in row major order
        flipped_transform.SetMatrix([1,0,0,-1])
    else:
        flipped_transform.SetMatrix([-1,0,0,0,1,0,0,0,1])
    centered_transform.AddTransform(flipped_transform)
    
    # Resample onto the reference image 
    flipped_image = sitk.Resample(img, reference_image, centered_transform, sitk.sitkLinear, 0.0)
    flipped_images.append(flipped_image)
    sitk.WriteImage(flipped_image,os.path.join(OUTPUT_DIR, 'spatial_aug'+str(index))+'_fliped_.nrrd')
    
# Uncomment the following line to display the images (we don't want to time this)
#disp_images(flipped_images, fig_size=(6,2))
'''

