import sys, os
import json
from itertools import compress
import torch

import numpy as np
import matplotlib.pyplot as plt
import scipy
import scipy.io
import cv2
from PIL import Image


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
        
def multidim_argmax(M):
    return np.unravel_index(M.argmax(), M.shape)

def normalize(M):
    """ Take all values of M and normalize it to the range [0,1]
    """
    M = M.astype(np.float32)
    return (M - M.min()) / (M.max() - M.min())

def resize_image(img, new_size, interpolation='zoom'):
    """ Resizes an image to new_size.

        Default interpolation uses cv2.INTER_LINEAR (good for zooming)
    """
    if interpolation == 'zoom':
        interp = cv2.INTER_LINEAR
    elif interpolation == 'shrink':
        interp = cv2.INTER_AREA
    elif interpolation == 'nearest':
        interp = cv2.INTER_NEAREST
    else:
        raise Exception("Interpolation should be one of: ['zoom', 'shrink', 'nearest']")
    return cv2.resize(img, new_size, interpolation=interp)

def load_rgb_image(imagefile):
    image = cv2.imread(imagefile)
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    return image

def load_rgb_image_with_resize(imagefile, new_size, interp='zoom'):
    """ Load image and resize
    """
    image = load_rgb_image(imagefile)
    image = resize_image(image, new_size, interpolation=interp)
    return image

def get_color_mask(object_index, nc=None):
    """ Colors each index differently. Useful for visualizing semantic masks

        @param object_index: a [H x W] numpy array of ints from {0, ..., nc-1}
        @param nc: total number of colors. If None, this will be inferred by masks

        @return: a [H x W x 3] RGB image of colored masks
    """
    object_index = object_index.astype(int)

    if nc is None:
        NUM_COLORS = object_index.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    color_mask = np.zeros(object_index.shape + (3,)).astype(np.uint8)
    for i in np.unique(object_index):
        if i == 0 or i == -1:
            continue
        color_mask[object_index == i, :] = np.array(colors[i][:3]) * 255
        
    return color_mask

def encode_one_hot_tensor(labels):
    """ Takes a torch tensor of integers and encodes it into a one-hot tensor.
        Let K be the number of labels

        @param labels: a [T x H x W] torch tensor with values in {0, ..., K-1}

        @return: a [T x K x H x W] torch tensor of 0's and 1's
    """
    T, H, W = labels.shape
    K = int(torch.max(labels).item() + 1)
    
    # Encode the one hot tensor
    one_hot_tensor = torch.zeros((T, K, H, W), device=labels.device)
    one_hot_tensor.scatter_(1, labels.long().unsqueeze(1), 1)

    return one_hot_tensor

def build_matrix_of_indices(height, width):
    """ Builds a [height, width, 2] numpy array containing coordinates.

        @return: 3d array B s.t. B[..., 0] contains y-coordinates, B[..., 1] contains x-coordinates
    """
    return np.indices((height, width), dtype=np.float32).transpose(1,2,0)

def torch_moi(h, w, device='cpu'):
    """ Torch function to do the same thing as above function, but returns CHW format
        Also, B[0, ...] is x-coordinates
        
        @return: 3d torch tensor B s.t. B[0, ...] contains x-coordinates, B[0, ...] contains y-coordinates
    """
    ys = torch.arange(h, device=device).view(-1,1).expand(h,w)
    xs = torch.arange(w, device=device).view(1,-1).expand(h,w)
    return torch.stack([xs,ys], dim=0).float()

def concatenate_spatial_coordinates(feature_map):
    """ Adds x,y coordinates as channels to feature map

        @param feature_map: a [T x C x H x W] torch tensor
    """
    T, C, H, W = feature_map.shape

    # build matrix of indices. then replicated it T times
    MoI = build_matrix_of_indices(H, W) # Shape: [H, W, 2]
    MoI = np.tile(MoI, (T, 1, 1, 1)) # Shape: [T, H, W, 2]
    MoI[..., 0] = MoI[..., 0] / (H-1) * 2 - 1 # in [-1, 1]
    MoI[..., 1] = MoI[..., 1] / (W-1) * 2 - 1
    MoI = torch.from_numpy(MoI).permute(0,3,1,2).to(feature_map.device) # Shape: [T, 2, H, W]

    # Concatenate on the channels dimension
    feature_map = torch.cat([feature_map, MoI], dim=1)

    return feature_map

def append_channels_dim(img):
    """ If an image is 2D (shape: [H x W]), add a channels dimensions to bring it to: [H x W x 1]

        This is to be called after cv2.resize, since if you resize a [H x W x 1] image, cv2.resize
            spits out a [new_H x new_W] image
    """
    if img.ndim == 2:
        # append axis dimension
        img = np.expand_dims(img, axis=-1)
        return img
    elif img.ndim == 3:
        # do nothing
        return img
    else:
        # wtf is happening
        raise Exception("This image is a weird shape: {0}".format(img.shape))


def visualize_segmentation(im, masks, nc=None, save_dir=None):
    """ Visualize segmentations nicely. Based on code from:
        https://github.com/roytseng-tw/Detectron.pytorch/blob/master/lib/utils/vis.py

        @param im: a [H x W x 3] RGB image. numpy array of dtype np.uint8
        @param masks: a [H x W] numpy array of dtype np.uint8 with values in {0, ..., K}
        @param nc: total number of colors. If None, this will be inferred by masks
    """ 
    from matplotlib.patches import Polygon

    masks = masks.astype(int)
    im = im.copy()

    # Generate color mask
    if nc is None:
        NUM_COLORS = masks.max() + 1
    else:
        NUM_COLORS = nc

    cm = plt.get_cmap('gist_rainbow')
    colors = [cm(1. * i/NUM_COLORS) for i in range(NUM_COLORS)]

    # Mask
    imgMask = np.zeros(im.shape)


    # Draw color masks
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Add to the mask
        imgMask[e] = color_mask

    # Add the mask to the image
    imgMask = (imgMask * 255).round().astype(np.uint8)
    im = cv2.addWeighted(im, 0.5, imgMask, 0.5, 0.0)


    # Draw mask contours
    for i in np.unique(masks):
        if i == 0: # background
            continue

        # Get the color mask
        color_mask = np.array(colors[i][:3])
        w_ratio = .4
        for c in range(3):
            color_mask[c] = color_mask[c] * (1 - w_ratio) + w_ratio
        e = (masks == i)

        # Find contours
        contour, hier = cv2.findContours(
            e.astype(np.uint8).copy(), cv2.RETR_CCOMP, cv2.CHAIN_APPROX_NONE)

        # Plot the nice outline
        for c in contour:
            cv2.drawContours(im, contour, -1, (255,255,255), 2)


    if save_dir is not None:
        # Save the image
        PIL_image = Image.fromarray(im)
        PIL_image.save(save_dir)
        return PIL_image
    else:
        return im
    

### These two functions were adatped from the DAVIS public dataset ###

def imread_indexed(filename):
    """ Load segmentation image (with palette) given filename."""
    im = Image.open(filename)
    annotation = np.array(im)
    return annotation

def imwrite_indexed(filename,array):
    """ Save indexed png with palette."""

    palette_abspath = '/data/tabletop_dataset_v5/palette.txt' # hard-coded filepath
    color_palette = np.loadtxt(palette_abspath, dtype=np.uint8).reshape(-1,3)

    if np.atleast_3d(array).shape[2] != 1:
        raise Exception("Saving indexed PNGs requires 2D array.")

    im = Image.fromarray(array)
    im.putpalette(color_palette.ravel())
    im.save(filename, format='PNG')

def seg2bmap(seg, return_contour=False):
    """ From a segmentation, compute a binary boundary map with 1 pixel wide
        boundaries. This boundary lives on the mask, i.e. it's a subset of the mask.

        @param seg: a [H x W] numpy array of values in {0,1}

        @return: a [H x W] numpy array of values in {0,1}
                 a [2 x num_boundary_pixels] numpy array. [0,:] is y-indices, [1,:] is x-indices
    """
    seg = seg.astype(np.uint8)
    contours, hierarchy = cv2.findContours(seg, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_NONE)
    temp = np.zeros_like(seg)
    bmap = cv2.drawContours(temp, contours, -1, 1, 1)

    if return_contour: # Return the SINGLE largest contour
        contour_sizes = [len(c) for c in contours]
        ind = np.argmax(contour_sizes)
        contour = np.ascontiguousarray(np.fliplr(contours[ind][:,0,:]).T) # Shape: [2 x num_boundary_pixels]
        return bmap, contour
    else:
        return bmap

def subplotter(images, max_plots_per_row=4, fig_index_start=1):
    """Plot images side by side.
    
    Args:
        images: an Iterable of [H, W, C] np.arrays. If images is
            a dictionary, the values are assumed to be the arrays,
            and the keys are strings which will be titles.
    """
    
    if type(images) not in [list, dict]:
        raise Exception("images MUST be type list or dict...")

    fig_index = fig_index_start
    
    num_plots = len(images)
    num_rows = int(np.ceil(num_plots / max_plots_per_row))

    for row in range(num_rows):

        fig = plt.figure(fig_index, figsize=(max_plots_per_row*5, 5))
        fig_index += 1

        for j in range(max_plots_per_row):

            ind = row*max_plots_per_row + j
            if ind >= num_plots:
                break

            plt.subplot(1, max_plots_per_row, j+1)
            if type(images) == dict:
                title = list(images.keys())[ind]
                image = images[title]
                plt.title(title)
            else:
                image = images[ind]
            plt.imshow(image)

def gallery(images, row_height='auto'):
    """Shows a set of images in a gallery that flexes with the width of the notebook.
    
    Args:
        images: an Iterable of [H, W, C] np.arrays. If images is
            a dictionary, the values are assumed to be the arrays,
            and the keys are strings which will be titles.

        row_height: str
            CSS height value to assign to all images. Set to 'auto' by default to show images
            with their native dimensions. Set to a value like '250px' to make all rows
            in the gallery equal height.
    """
    def _src_from_data(data):
        """Base64 encodes image bytes for inclusion in an HTML img element"""
        img_obj = IP_Image(data=data)
        for bundle in img_obj._repr_mimebundle_():
            for mimetype, b64value in bundle.items():
                if mimetype.startswith('image/'):
                    return f'data:{mimetype};base64,{b64value}'

    def _get_img_as_bytestring(img):
        im = Image.fromarray(img)
        buf = io.BytesIO()
        im.save(buf, format='JPEG')
        return buf.getvalue()
    
    if type(images) not in [list, dict]:
        raise Exception("images MUST be type list or dict...")
    
    num_images = len(images)
    
    figures = []
    for i in range(num_images):
        if isinstance(images, list):
            caption = ''
            image = images[i]
        else: # dict
            caption = list(images.keys())[i]
            image = images[caption]
        src = _src_from_data(_get_img_as_bytestring(image))

        figures.append(f'''
            <figure style="margin: 5px !important;">
              <img src="{src}" style="height: {row_height}">
              {caption}
            </figure>
        ''')
        
    IP_display(IP_HTML(data=f'''
        <div style="display: flex; flex-flow: row wrap; text-align: center;">
        {''.join(figures)}
        </div>
    '''))