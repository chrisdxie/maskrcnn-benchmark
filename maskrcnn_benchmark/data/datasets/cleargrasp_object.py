import torch
from torch.utils.data import Dataset, DataLoader
import os
import numpy as np
import cv2
import glob
import yaml
import json
from easydict import EasyDict as edict
import OpenEXR, Imath
from pathlib import Path
from scipy.ndimage.measurements import label as connected_components

# My libraries
from maskrcnn_benchmark.data.datasets import data_augmentation
from maskrcnn_benchmark.data.datasets import util as util_




def worker_init_fn(worker_id):
    """ Use this to bypass issue with PyTorch dataloaders using deterministic RNG for Numpy
        https://github.com/pytorch/pytorch/issues/5059
    """
    np.random.seed(np.random.get_state()[1][0] + worker_id)

def exr_loader(EXR_PATH, ndim=3):
    """Loads a .exr file as a numpy array
    Args:
        EXR_PATH: path to the exr file
        ndim: number of channels that should be in returned array. Valid values are 1 and 3.
                        if ndim=1, only the 'R' channel is taken from exr file
                        if ndim=3, the 'R', 'G' and 'B' channels are taken from exr file.
                            The exr file must have 3 channels in this case.
    Returns:
        numpy.ndarray (dtype=np.float32): If ndim=1, shape is (height x width)
                                          If ndim=3, shape is (3 x height x width)
    """

    exr_file = OpenEXR.InputFile(EXR_PATH)
    cm_dw = exr_file.header()['dataWindow']
    size = (cm_dw.max.x - cm_dw.min.x + 1, cm_dw.max.y - cm_dw.min.y + 1)

    pt = Imath.PixelType(Imath.PixelType.FLOAT)

    if ndim == 3:
        # read channels indivudally
        allchannels = []
        for c in ['R', 'G', 'B']:
            # transform data to numpy
            channel = np.frombuffer(exr_file.channel(c, pt), dtype=np.float32)
            channel.shape = (size[1], size[0])
            allchannels.append(channel)

        # create array and transpose dimensions to match tensor style
        exr_arr = np.array(allchannels).transpose((0, 1, 2))
        return exr_arr

    if ndim == 1:
        # transform data to numpy
        channel = np.frombuffer(exr_file.channel('R', pt), dtype=np.float32)
        channel.shape = (size[1], size[0])  # Numpy arrays are (row, col)
        exr_arr = np.array(channel)
        return exr_arr


def compute_xyz(depth_img, camera_params):
    """ Compute ordered point cloud from depth image and camera parameters.

        If focal lengths fx,fy are stored in the camera_params dictionary, use that.
        Else, assume camera_params contains parameters used to generate synthetic data (e.g. fov, near, far, etc)

        @param depth_img: a [H x W] numpy array of depth values in meters
        @param camera_params: a dictionary with parameters of the camera used 
    """

    # Compute focal length from camera parameters
    fx = camera_params['fx']
    fy = camera_params['fy']
    x_offset = camera_params['cx']
    y_offset = camera_params['cy']
    indices = util_.build_matrix_of_indices(camera_params['yres'], camera_params['xres'])
    indices[..., 0] = np.flipud(indices[..., 0]) # pixel indices start at top-left corner. for these equations, it starts at bottom-left

    z_e = depth_img
    x_e = (indices[..., 1] - x_offset) * z_e / fx
    y_e = (indices[..., 0] - y_offset) * z_e / fy
    xyz_img = np.stack([x_e, y_e, z_e], axis=-1) # Shape: [H x W x 3]
    return xyz_img

def filter_labels(labels, bboxes):
    labels_new = labels.clone()
    height = labels.shape[1]
    width = labels.shape[2]
    for i in range(labels.shape[0]):
        label = labels[i]
        bbox = bboxes[i].cpu().numpy()

        bbox_mask = torch.zeros_like(label)
        for j in range(bbox.shape[0]):
            x1 = max(int(bbox[j, 0]), 0)
            y1 = max(int(bbox[j, 1]), 0)
            x2 = min(int(bbox[j, 2]), width-1)
            y2 = min(int(bbox[j, 3]), height-1)
            bbox_mask[y1:y2, x1:x2] = 1

        mask_ids = torch.unique(label)
        if mask_ids[0] == 0:
            mask_ids = mask_ids[1:]

        for index, mask_id in enumerate(mask_ids):
            mask = (label == mask_id).float()
            percentage = torch.sum(mask * bbox_mask) / torch.sum(mask)
            if percentage > 0.8:
                labels_new[i][label == mask_id] = 0

    return labels_new

class ClearGraspDataset(Dataset):
    def __init__(self, image_set, cleargrasp_object_path = None):
        """
            @param image_set: a string in ['train', 'test']
        """

        self.name = 'cleargrasp_object_' + image_set
        self._image_set = image_set
        self._cleargrasp_object_path = self._get_default_path() if cleargrasp_object_path is None \
                            else cleargrasp_object_path
        self._width = 640
        self._height = 480
        self.image_paths, self.mask_paths, self.depth_paths, self.camera_intrinsics, self.opaque_annotations = self.list_dataset()

        print('%d images for dataset %s' % (len(self.image_paths), self.name))
        self._size = len(self.image_paths)
        assert os.path.exists(self._cleargrasp_object_path), \
                'cleargrasp_object path does not exist: {}'.format(self._cleargrasp_object_path)


    def list_dataset(self):

        data_path = []
        if self._image_set == 'train':
            data_path.append(os.path.join(self._cleargrasp_object_path, 'cleargrasp-dataset-train')) # I don't have this
        else:
            data_path.append(os.path.join(self._cleargrasp_object_path, 'cleargrasp-dataset-test-val', 'real-test'))
            data_path.append(os.path.join(self._cleargrasp_object_path, 'cleargrasp-dataset-test-val', 'real-val'))

        image_paths = []
        mask_paths = []
        depth_paths = []
        opaque_annotations = []
        camera_intrinsics = {}
        for i in range(len(data_path)):
            for camera in ['d415', 'd435']:
                dirpath = Path(os.path.join(data_path[i], camera))
                if not dirpath.exists():
                    continue

                image_paths += sorted(list(dirpath.glob('*-transparent-rgb-img.jpg')))
                mask_paths += sorted(list(dirpath.glob('*-mask.png')))
                depth_paths += sorted(list(dirpath.glob('*-transparent-depth-img.exr')))

                # camera intrinsics
                filename = os.path.join(data_path[i], camera, 'camera_intrinsics.yaml')
                with open(filename, 'r') as f:
                    intrinsics = edict(yaml.load(f))
                camera_intrinsics[camera] = intrinsics

                # object bounding boxes of opaque objects
                with (dirpath / 'opaque_objects.json').open('r') as f:
                    annotations = json.load(f)
                    opaque_annotations += annotations

        return image_paths, mask_paths, depth_paths, camera_intrinsics, opaque_annotations

    def __len__(self):
        return self._size


    def _get_default_path(self):
        """
        Return the default path where cleargrasp_object is expected to be installed.
        """
        return os.path.join('/data', 'cleargrasp')

    def process_label(self, foreground_labels):
        """ Process foreground_labels
                - Map the foreground_labels to {0, 1, ..., K-1}

            @param foreground_labels: a [H x W] numpy array of labels

            @return: foreground_labels
        """
        # Find the unique (nonnegative) foreground_labels, map them to {0, ..., K-1}
        unique_nonnegative_indices = np.unique(foreground_labels)
        mapped_labels = foreground_labels.copy()
        for k in range(unique_nonnegative_indices.shape[0]):
            mapped_labels[foreground_labels == unique_nonnegative_indices[k]] = k
        foreground_labels = mapped_labels
        return foreground_labels


    def __getitem__(self, idx):

        # BGR image
        filename = str(self.image_paths[idx])
        rgb_img = cv2.cvtColor(cv2.imread(filename), cv2.COLOR_BGR2RGB)

        # rescale width to 640
        width = rgb_img.shape[1]
        im_scale = 640.0 / width

        if im_scale != 1.0:
            rgb_img = util_.resize_image(rgb_img, (640,368), interpolation='zoom')

        # find the opaque annotation
        for i in range(len(self.opaque_annotations)):
            ann = self.opaque_annotations[i]
            start = ann['filename'].find('cleargrasp')
            name = ann['filename'][start:]
            if name in filename:
                boxes = ann['annotations']
                num = len(boxes)
                bbox = np.zeros((num, 4), dtype=np.float32)
                for j in range(num):
                    bbox[j, 0] = boxes[j]['x']
                    bbox[j, 1] = boxes[j]['y']
                    bbox[j, 2] = boxes[j]['x'] + boxes[j]['width']
                    bbox[j, 3] = boxes[j]['y'] + boxes[j]['height']
                bbox = bbox * im_scale
                break

        # Label
        labels_filename = str(self.mask_paths[idx])
        mask = util_.imread_indexed(labels_filename)
        if len(mask.shape) == 3:
            mask = mask[:, :, 0]
        foreground_labels, num_components = connected_components(mask == 255)
        foreground_labels = self.process_label(foreground_labels)
        if im_scale != 1.0:
            foreground_labels = util_.resize_image(foreground_labels, (640,368), interpolation='nearest')
        label_blob = torch.from_numpy(foreground_labels)#.unsqueeze(0)
        label_blob[label_blob > 0] = label_blob[label_blob > 0] + 1 # so values are in [0, 2, 3, ...] (e.g. no table label)

        # Label path
        temp_idx = labels_filename.split('/').index('cleargrasp') # parse something like this: /data/cleargrasp/cleargrasp-dataset-test-val/real-test/d415/000000035-mask.png
        label_abs_path = '/'.join(labels_filename.split('/')[temp_idx+1:])

        # Depth image
        depth_filename = str(self.depth_paths[idx])
        depth = exr_loader(depth_filename, 1)
        if 'd415' in depth_filename:
            camera_params = self.camera_intrinsics['d415']
        else:
            camera_params = self.camera_intrinsics['d435']
        xyz_img = compute_xyz(depth, camera_params)
        if im_scale != 1.0:
            xyz_img = util_.resize_image(xyz_img, (640,368), interpolation='nearest')
        depth_blob = torch.from_numpy(xyz_img).permute(2, 0, 1)

        sample = {'rgb': rgb_img, # Shape: [H x W x 3]
                  'xyz': depth_blob,
                  'labels': label_blob,
                  'bbox': torch.from_numpy(bbox),
                  'label_abs_path' : label_abs_path,
                  }

        return sample


def get_CG_dataloader(batch_size=8, num_workers=4, shuffle=True):

    dataset = ClearGraspDataset('test')

    return DataLoader(dataset=dataset,
                      batch_size=batch_size,
                      shuffle=shuffle,
                      num_workers=num_workers,
                      worker_init_fn=worker_init_fn)
