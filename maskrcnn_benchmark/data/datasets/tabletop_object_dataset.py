# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
import torch
import torchvision
import cv2
import glob
import numpy as np

from torch.utils.data import Dataset, DataLoader

from maskrcnn_benchmark.structures.bounding_box import BoxList
from maskrcnn_benchmark.structures.segmentation_mask import SegmentationMask

# My libraries
from maskrcnn_benchmark.data.datasets import data_augmentation
from maskrcnn_benchmark.data.datasets import util as util_


data_loading_params = {
    # Camera/Frustum parameters
    'img_width' : 640, 
    'img_height' : 480,
    'near' : 0.01,
    'far' : 100,
    'fov' : 45, # vertical field of view in angles
    
    'use_data_augmentation' : True,

    # Multiplicative noise
    'gamma_shape' : 1000.,
    'gamma_scale' : 0.001,
    
    # Additive noise
    'gaussian_scale' : 0.005, # 1cm standard dev
    'gp_rescale_factor' : 4,
    
    # Random ellipse dropout
    'ellipse_dropout_mean' : 10, 
    'ellipse_gamma_shape' : 5.0, 
    'ellipse_gamma_scale' : 1.0,

    # Random high gradient dropout
    'gradient_dropout_left_mean' : 15, 
    'gradient_dropout_alpha' : 2., 
    'gradient_dropout_beta' : 5.,

    # Random pixel dropout
    'pixel_dropout_alpha' : 1., 
    'pixel_dropout_beta' : 10.,
    
}
data_dir = '/data/tabletop_dataset_v5/training_set/'



def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    if 'fx' in camera_params and 'fy' in camera_params:
        fx = camera_params['fx']
        fy = camera_params['fy']
    else: # simulated data
        aspect_ratio = camera_params['img_width'] / camera_params['img_height']
        e = 1 / (np.tan(np.radians(camera_params['fov']/2.)))
        t = camera_params['near'] / e; b = -t
        r = t * aspect_ratio; l = -r
        alpha = camera_params['img_width'] / (r-l) # pixels per meter
        focal_length = camera_params['near'] * alpha # focal length of virtual camera (frustum camera)
        fx = focal_length; fy = focal_length

    if 'x_offset' in camera_params and 'y_offset' in camera_params:
        x_offset = camera_params['x_offset']
        y_offset = camera_params['y_offset']
    else: # simulated data
        x_offset = camera_params['img_width']/2
        y_offset = camera_params['img_height']/2

    indices = util_.build_matrix_of_indices(camera_params['img_height'], camera_params['img_width'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for OpenGL, it starts at bottom-left
    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    
    return xyz_img

def mask_to_tight_box(mask):
    a = np.transpose(np.nonzero(mask)) 
    bbox = np.min(a[:, 1]), np.min(a[:, 0]), np.max(a[:, 1]), np.max(a[:, 0])
    return bbox  # x_min, y_min, x_max, y_max


NUM_VIEWS_PER_SCENE = 6 # background only is not trained on
class Tabletop_Object_Dataset(Dataset):

    def __init__(self, transforms=None):
        # don't use transforms. that's just to satisfy the API
        self.base_dir = data_dir
        self.params = data_loading_params.copy()

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop'

        # This is not used
        self.classid_to_name = {
            0 : "background",
            1 : "table",
            2 : "object"
        }

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
        """
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = data_augmentation.BGR_image(rgb_img)
        rgb_img = data_augmentation.array_to_tensor(rgb_img)

        return rgb_img

    def process_depth(self, depth_img, seg_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)

        # add random noise to depth
        if self.params['use_data_augmentation']:
            depth_img = data_augmentation.add_noise_to_depth(depth_img, self.params)
            depth_img = data_augmentation.dropout_random_ellipses(depth_img, self.params)
            # depth_img = data_augmentation.dropout_near_high_gradients(depth_img, seg_img, self.params)
            # depth_img = data_augmentation.dropout_random_pixels(depth_img, self.params)

        # Compute xyz ordered point cloud
        xyz_img = compute_xyz(depth_img, self.params)
        if self.params['use_data_augmentation']:
            xyz_img = data_augmentation.add_noise_to_xyz(xyz_img, depth_img, self.params)

        xyz_img = data_augmentation.array_to_tensor(xyz_img)

        return xyz_img

    def process_label(self, labels):
        """ Process labels
                - Map the labels to [H x W x num_instances] numpy array
        """
        H, W = labels.shape

        # Find the unique (nonnegative) labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(labels)

        # Drop 0 if it's in there
        if 0 == unique_nonnegative_indices[0]:
            unique_nonnegative_indices = unique_nonnegative_indices[1:]
        num_instances = unique_nonnegative_indices.shape[0]
        # NOTE: IMAGES WITH BACKGROUND ONLY HAVE NO INSTANCES

        # Get binary masks
        binary_masks = np.zeros((H, W, num_instances), dtype=np.float32)
        for i, label in enumerate(unique_nonnegative_indices):
            binary_masks[..., i] = (labels == label).astype(np.float32)

        # Get bounding boxes
        boxes = np.zeros((num_instances, 4))
        for i in range(num_instances):
            boxes[i, :] = np.array(mask_to_tight_box(binary_masks[..., i]))

        # Get labels for each mask
        labels = unique_nonnegative_indices.clip(1,2)

        # Turn them into torch tensors
        boxes = data_augmentation.array_to_tensor(boxes)
        binary_masks = data_augmentation.array_to_tensor(binary_masks)
        labels = data_augmentation.array_to_tensor(labels).long()

        return boxes, binary_masks, labels

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]

        # Get view number
        view_num = idx % NUM_VIEWS_PER_SCENE + 1 # view_num=0 is always background with no table/objects

        # Label
        seg_img_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        seg_img = util_.imread_indexed(seg_img_filename)
        boxes, binary_masks, labels = self.process_label(seg_img)
        # boxes.shape: [num_instances x 4], binary_masks.shape: [num_instances x H x W], labels.shape: [num_instances]

        # RGB image
        if self.params['use_rgb'] and not self.params['use_depth']:
            rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
            rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
            img = self.process_rgb(rgb_img) # Shape: [3 x H x W]

        # Depth image
        if self.params['use_depth'] and not self.params['use_rgb']:
            depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
            depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
            img = self.process_depth(depth_img, labels) # Shape: [3 x H x W]

        # RGB + Depth
        if self.params['use_depth'] and self.params['use_rgb']:
            rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
            rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
            rgb_img = self.process_rgb(rgb_img) # Shape: [3 x H x W]

            depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
            depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
            depth_img = self.process_depth(depth_img, labels) # Shape: [3 x H x W]

            # Concatenate these
            img = torch.cat([rgb_img, depth_img], dim=0)


        # Create BoxList stuff
        target = BoxList(boxes, (self.params['img_width'], self.params['img_height']), mode="xyxy")
        target.add_field("labels", labels)
        masks = SegmentationMask(binary_masks, (self.params['img_width'], self.params['img_height']), "mask")
        target.add_field("masks", masks)

        return img, target, idx


    def get_img_info(self, idx):
        # get img_height and img_width. This is used if
        # we want to split the batches according to the aspect ratio
        # of the image, as it can be more efficient than loading the
        # image from disk
        return {"height": self.params['img_height'], 
                "width": self.params['img_width'],
                "idx": idx
               }


class Tabletop_Object_Dataset_Depth(Tabletop_Object_Dataset):

    def __init__(self, transforms=None):
        # don't use transforms. that's just to satisfy the API
        self.base_dir = data_dir
        self.params = data_loading_params.copy()
        self.params['use_rgb'] = False
        self.params['use_depth'] = True

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop_Depth'

        # This is not used
        self.classid_to_name = {
            0 : "background",
            1 : "table",
            2 : "object"
        }


class Tabletop_Object_Dataset_RGB(Tabletop_Object_Dataset):

    def __init__(self, transforms=None):
        # don't use transforms. that's just to satisfy the API
        self.base_dir = data_dir
        self.params = data_loading_params.copy()
        self.params['use_rgb'] = True
        self.params['use_depth'] = False

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop_RGB'

        # This is not used
        self.classid_to_name = {
            0 : "background",
            1 : "table",
            2 : "object"
        }

class Tabletop_Object_Dataset_RGBD(Tabletop_Object_Dataset):

    def __init__(self, transforms=None):
        # don't use transforms. that's just to satisfy the API
        self.base_dir = data_dir
        self.params = data_loading_params.copy()
        self.params['use_rgb'] = True
        self.params['use_depth'] = True

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * NUM_VIEWS_PER_SCENE

        self.name = 'TableTop_RGB'

        # This is not used
        self.classid_to_name = {
            0 : "background",
            1 : "table",
            2 : "object"
        }



##### Separate Class for testing #####

class Tabletop_Object_Dataset_Test(Dataset):
    """ Data loader for Tabletop Object Dataset
    """
    NUM_VIEWS_PER_SCENE = 5


    def __init__(self, base_dir):
        self.base_dir = base_dir

        # Get a list of all scenes
        self.scene_dirs = sorted(glob.glob(self.base_dir + '*/'))
        self.len = len(self.scene_dirs) * self.NUM_VIEWS_PER_SCENE

        self.name = 'TableTop'

        if 'v6' in self.base_dir:
            global OBJECTS_LABEL
            OBJECTS_LABEL = 4

    def __len__(self):
        return self.len

    def process_rgb(self, rgb_img):
        """ Process RGB image
        """
        rgb_img = rgb_img.astype(np.float32)
        rgb_img = data_augmentation.BGR_image(rgb_img)
        rgb_img = data_augmentation.array_to_tensor(rgb_img)

        return rgb_img

    def process_depth(self, depth_img):
        """ Process depth channel
                TODO: CHANGE THIS
                - change from millimeters to meters
                - cast to float32 data type
                - add random noise
                - compute xyz ordered point cloud
        """

        # millimeters -> meters
        depth_img = (depth_img / 1000.).astype(np.float32)
        xyz_img = compute_xyz(depth_img, data_loading_params)
        xyz_img = data_augmentation.array_to_tensor(xyz_img)

        return xyz_img

    def __getitem__(self, idx):

        cv2.setNumThreads(0) # some hack to make sure pyTorch doesn't deadlock. Found at https://github.com/pytorch/pytorch/issues/1355. Seems to work for me

        # Get scene directory
        scene_idx = idx // self.NUM_VIEWS_PER_SCENE
        scene_dir = self.scene_dirs[scene_idx]
        view_num = idx % self.NUM_VIEWS_PER_SCENE + 2

        # RGB image
        rgb_img_filename = scene_dir + f"rgb_{view_num:05d}.jpeg"
        rgb_img = cv2.cvtColor(cv2.imread(rgb_img_filename), cv2.COLOR_BGR2RGB)
        rgb_img = self.process_rgb(rgb_img)

        # Depth image
        depth_img_filename = scene_dir + f"depth_{view_num:05d}.png"
        depth_img = cv2.imread(depth_img_filename, cv2.IMREAD_ANYDEPTH) # This reads a 16-bit single-channel image. Shape: [H x W]
        xyz_img = self.process_depth(depth_img)

        foreground_labels_filename = scene_dir + f"segmentation_{view_num:05d}.png"
        label_abs_path = '/'.join(foreground_labels_filename.split('/')[-2:]) # Used for evaluation

        return {'rgb' : rgb_img,
                'xyz' : xyz_img,
                'label_abs_path' : label_abs_path,
                }

def TOD_test_dataloader(base_dir, batch_size=8, num_workers=4, shuffle=False):

    return DataLoader(dataset=Tabletop_Object_Dataset_Test(base_dir),
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers)

