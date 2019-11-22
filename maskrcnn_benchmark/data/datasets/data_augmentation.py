import torch
import random
import numpy as np
import numbers
import torchvision.transforms as transforms
from PIL import Image # PyTorch likes PIL instead of cv2
import cv2

def array_to_tensor(array):
    """ Converts a numpy.ndarray (N x H x W x C) to a torch.FloatTensor of shape (N x C x H x W)
        OR
        converts a nump.ndarray (H x W x C) to a torch.FloatTensor of shape (C x H x W)
    """

    if array.ndim == 4: # NHWC
        tensor = torch.from_numpy(array).permute(0,3,1,2).float()
    elif array.ndim == 3: # HWC
        tensor = torch.from_numpy(array).permute(2,0,1).float()
    else: # everything else
        tensor = torch.from_numpy(array).float()

    return tensor

##### RGB transformations #####

def standardize_image(image):
    """ Converts numpy.ndarray [H x W x 3] of images to [0,1] range, and then standardizes
    """
    image_standardized = np.zeros_like(image)

    mean=[0.485, 0.456, 0.406]
    std=[0.229, 0.224, 0.225]
    for i in range(3):
        image_standardized[...,i] = (image[...,i]/255 - mean[i]) / std[i]

    return image_standardized

def BGR_image(image):
    """ Converts an RGB image (np.uint8, [0, 255]) of size [H x W x 3] to a BGR image in [0,1]
    """
    image = image.astype(np.float32)
    image = image / 255.
    image = image[..., np.array([2,1,0])]

    return image

def random_color_warp(image, d_h=None, d_s=None, d_l=None):
    """ Given an RGB image [H x W x 3], add random hue, saturation and luminosity to the image

        Code adapted from: https://github.com/yuxng/PoseCNN/blob/master/lib/utils/blob.py
    """
    H, W, _ = image.shape

    image_color_warped = np.zeros_like(image)

    # Set random hue, luminosity and saturation which ranges from -0.1 to 0.1
    if d_h is None:
        d_h = (random.random() - 0.5) * 0.2 * 256
    if d_l is None:
        d_l = (random.random() - 0.5) * 0.2 * 256
    if d_s is None:
        d_s = (random.random() - 0.5) * 0.2 * 256

    # Convert the RGB to HLS
    hls = cv2.cvtColor(image.round().astype(np.uint8), cv2.COLOR_RGB2HLS)
    h, l, s = cv2.split(hls)

    # Add the values to the image H, L, S
    new_h = (np.round((h + d_h)) % 256).astype(np.uint8)
    new_l = np.round(np.clip(l + d_l, 0, 255)).astype(np.uint8)
    new_s = np.round(np.clip(s + d_s, 0, 255)).astype(np.uint8)

    # Convert the HLS to RGB
    new_hls = cv2.merge((new_h, new_l, new_s)).astype(np.uint8)
    new_im = cv2.cvtColor(new_hls, cv2.COLOR_HLS2RGB)

    image_color_warped = new_im.astype(np.float32)

    return image_color_warped

##### Depth transformations #####

def add_noise_to_depth(depth_img, noise_params):
    """ Add noise to depth image. 
        This is adapted from the DexNet 2.0 code.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Multiplicative noise: Gamma random variable
    multiplicative_noise = np.random.gamma(noise_params['gamma_shape'], noise_params['gamma_scale'])
    depth_img = multiplicative_noise * depth_img

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    # small_H, small_W = (np.array(depth_img.shape) / noise_params['gp_rescale_factor']).astype(int)
    # additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W))
    # additive_noise = cv2.resize(additive_noise, depth_img.shape[::-1], interpolation=cv2.INTER_CUBIC)
    # depth_img[depth_img > 0] += additive_noise[depth_img > 0]

    return depth_img

def add_noise_to_xyz(xyz_img, depth_img, noise_params):
    """ Add (approximate) Gaussian Process noise to ordered point cloud

        @param xyz_img: a [H x W x 3] ordered point cloud
    """
    xyz_img = xyz_img.copy()

    H, W, C = xyz_img.shape

    # Additive noise: Gaussian process, approximated by zero-mean anisotropic Gaussian random variable,
    #                 which is rescaled with bicubic interpolation.
    small_H, small_W = (np.array([H, W]) / noise_params['gp_rescale_factor']).astype(int)
    additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(small_H, small_W, C))
    additive_noise = cv2.resize(additive_noise, (W, H), interpolation=cv2.INTER_CUBIC)
    # additive_noise = np.random.normal(loc=0.0, scale=noise_params['gaussian_scale'], size=(H,W,C))
    xyz_img[depth_img > 0, :] += additive_noise[depth_img > 0, :]

    return xyz_img

def dropout_random_ellipses(depth_img, noise_params):
    """ Randomly drop a few ellipses in the image for robustness.
        This is adapted from the DexNet 2.0 code.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample number of ellipses to dropout
    num_ellipses_to_dropout = np.random.poisson(noise_params['ellipse_dropout_mean'])

    # Sample ellipse centers
    nonzero_pixel_indices = np.array(np.where(depth_img > 0)).T # Shape: [#nonzero_pixels x 2]
    dropout_centers_indices = np.random.choice(nonzero_pixel_indices.shape[0], size=num_ellipses_to_dropout)
    dropout_centers = nonzero_pixel_indices[dropout_centers_indices, :] # Shape: [num_ellipses_to_dropout x 2]

    # Sample ellipse radii and angles
    x_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    y_radii = np.random.gamma(noise_params['ellipse_gamma_shape'], noise_params['ellipse_gamma_scale'], size=num_ellipses_to_dropout)
    angles = np.random.randint(0, 360, size=num_ellipses_to_dropout)

    # Dropout ellipses
    for i in range(num_ellipses_to_dropout):
        center = dropout_centers[i, :]
        x_radius = np.round(x_radii[i]).astype(int)
        y_radius = np.round(y_radii[i]).astype(int)
        angle = angles[i]

        # dropout the ellipse
        mask = np.zeros_like(depth_img)
        mask = cv2.ellipse(mask, tuple(center[::-1]), (x_radius, y_radius), angle=angle, startAngle=0, endAngle=360, color=1, thickness=-1)
        depth_img[mask == 1] = 0

    return depth_img

def dropout_near_high_gradients(depth_img, seg_img, noise_params):
    """ Randomly dropout areas near high gradients. "High gradients" are determined by segmentation mask edges
        This is adapted from the DexNet 2.0 code.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Calculate segmentation mask edges
    mask = seg_img != np.roll(seg_img, -1, axis=0)
    mask = np.logical_or(mask, seg_img != np.roll(seg_img, 1, axis=0))
    mask = np.logical_or(mask, seg_img != np.roll(seg_img, -1, axis=1))
    mask = np.logical_or(mask, seg_img != np.roll(seg_img, 1, axis=1))

    # Randomly choose how far left to go
    num_pixels_left = np.random.poisson(noise_params['gradient_dropout_left_mean'])
    for i in range(num_pixels_left):
        mask = np.logical_or(mask, np.roll(mask, -1, axis=0))

    # Randomly drop pixels from this mask
    mask_indices = np.array(np.where(mask)).T # Shape: [#nonzero_pixels x 2]
    percentage_of_mask_pixels_to_drop = np.random.beta(noise_params['gradient_dropout_alpha'], 
                                                       noise_params['gradient_dropout_beta'])
    num_pixels_to_drop = mask_indices.shape[0] * percentage_of_mask_pixels_to_drop
    num_pixels_to_drop = np.round(num_pixels_to_drop).astype(int)
    drop_indices = np.random.choice(mask_indices.shape[0], size=num_pixels_to_drop)
    depth_img[mask_indices[drop_indices, 0], mask_indices[drop_indices, 1]] = 0

    return depth_img

def dropout_random_pixels(depth_img, noise_params):
    """ Randomly dropout pixels in the depth image
        This is adapted from the DexNet 2.0 code.
        Their code: https://github.com/BerkeleyAutomation/gqcnn/blob/75040b552f6f7fb264c27d427b404756729b5e88/gqcnn/sgd_optimizer.py

        @param depth_img: a [H x W] set of depth z values
    """
    depth_img = depth_img.copy()

    # Sample how many pixels to drop
    drop_prob = np.random.beta(noise_params['pixel_dropout_alpha'],
                               noise_params['pixel_dropout_beta'])

    # Drop pixels
    drop_mask = np.random.choice(a=[False, True], size=depth_img.shape, p=[1-drop_prob, drop_prob])
    depth_img[drop_mask] = 0

    return depth_img

def blur_depth_img(depth_img, seg_img, noise_params):
    """ Gaussian blur depth image
    """
    depth_img = depth_img.copy()
    
    object_mask = seg_img.clip(0,2) == 2
    if np.count_nonzero(object_mask) == 0: # no objects, do nothing
        return depth_img

    blurred_depth_img = cv2.GaussianBlur(depth_img, (3,3), 0)
    depth_img[object_mask] = blurred_depth_img[object_mask]
    return depth_img