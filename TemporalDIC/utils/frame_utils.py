import numpy as np
from PIL import Image
from os.path import *
import re

import cv2
from torchvision.transforms import ColorJitter
cv2.setNumThreads(0)
cv2.ocl.setUseOpenCL(False)

TAG_CHAR = np.array([202021.25], np.float32)

def readFlow(fn):
    """ Read .flo file in Middlebury format"""

    with open(fn, 'rb') as f:
        magic = np.fromfile(f, np.float32, count=1)
        if 202021.25 != magic:
            print('Magic number incorrect. Invalid .flo file')
            return None
        else:
            w = np.fromfile(f, np.int32, count=1)
            h = np.fromfile(f, np.int32, count=1)
            data = np.fromfile(f, np.float32, count=2*int(w)*int(h))
            return np.resize(data, (int(h), int(w), 2))

def readPFM(file):
    file = open(file, 'rb')

    color = None
    width = None
    height = None
    scale = None
    endian = None

    header = file.readline().rstrip()
    if header == b'PF':
        color = True
    elif header == b'Pf':
        color = False
    else:
        raise Exception('Not a PFM file.')

    dim_match = re.match(rb'^(\d+)\s(\d+)\s$', file.readline())
    if dim_match:
        width, height = map(int, dim_match.groups())
    else:
        raise Exception('Malformed PFM header.')

    scale = float(file.readline().rstrip())
    if scale < 0: # little-endian
        endian = '<'
        scale = -scale
    else:
        endian = '>' # big-endian

    data = np.fromfile(file, endian + 'f')
    shape = (height, width, 3) if color else (height, width)

    data = np.reshape(data, shape)
    data = np.flipud(data)
    return data

def writeFlow(filename,uv,v=None):
    """ Write optical flow to file.
    
    If v is None, uv is assumed to contain both u and v channels,
    stacked in depth.
    Original code by Deqing Sun, adapted from Daniel Scharstein.
    """
    nBands = 2

    if v is None:
        assert(uv.ndim == 3)
        assert(uv.shape[2] == 2)
        u = uv[:,:,0]
        v = uv[:,:,1]
    else:
        u = uv

    assert(u.shape == v.shape)
    height,width = u.shape
    f = open(filename,'wb')
    # write the header
    f.write(TAG_CHAR)
    np.array(width).astype(np.int32).tofile(f)
    np.array(height).astype(np.int32).tofile(f)
    # arrange into matrix form
    tmp = np.zeros((height, width*nBands))
    tmp[:,np.arange(width)*2] = u
    tmp[:,np.arange(width)*2 + 1] = v
    tmp.astype(np.float32).tofile(f)
    f.close()


def readFlowKITTI(filename):
    flow = cv2.imread(filename, cv2.IMREAD_ANYDEPTH|cv2.IMREAD_COLOR)
    if flow is None:
        print(f"Failed to load image from file: {filename}")
        return None, None
    flow = flow[:,:,::-1].astype(np.float32)
    flow, valid = flow[:, :, :2], flow[:, :, 2]
    flow = (flow - 2**15) / 64.0
    print("File path for flow image:", filename)

    return flow, valid

def readDispKITTI(filename):
    disp = cv2.imread(filename, cv2.IMREAD_ANYDEPTH) / 256.0
    valid = disp > 0.0
    flow = np.stack([-disp, np.zeros_like(disp)], -1)
    return flow, valid


def writeFlowKITTI(filename, uv):
    uv = 64.0 * uv + 2**15
    valid = np.ones([uv.shape[0], uv.shape[1], 1])
    uv = np.concatenate([uv, valid], axis=-1).astype(np.uint16)
    cv2.imwrite(filename, uv[..., ::-1])
    

def read_gen(file_name, pil=False):
    ext = splitext(file_name)[-1]
    if ext == '.png' or ext == '.jpeg' or ext == '.ppm' or ext == '.jpg':
        return Image.open(file_name)
    elif ext == '.bin' or ext == '.raw':
        return np.load(file_name)
    elif ext == '.flo':
        return readFlow(file_name).astype(np.float32)
    elif ext == '.pfm':
        flow = readPFM(file_name).astype(np.float32)
        if len(flow.shape) == 2:
            return flow
        else:
            return flow[:, :, :-1]
    return []


class FlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.2, do_flip=True):

        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.5 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):
        """ Photometric augmentation """

        # asymmetric
        if np.random.rand() < self.asymmetric_color_aug_prob:
            imgs = [np.array(self.photo_aug(Image.fromarray(img)), dtype=np.uint8) for img in imgs]

        # symmetric
        else:
            img_num = len(imgs)
            image_stack = np.concatenate(imgs, axis=0)
            image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
            imgs = np.split(image_stack, img_num, axis=0)

        return imgs

    def eraser_transform(self, imgs, bounds=[50, 100]):
        print("[erasing]")
        ht, wd = imgs[0].shape[:2]
        if np.random.rand() < self.eraser_aug_prob:
            for idx in range(len(imgs)):
                mean_color = np.mean(imgs[idx].reshape(-1, 3), axis=0)
                for _ in range(np.random.randint(1, 3)):
                    # print("!@#!@#!@#!@#!@#!@#!")
                    x0 = np.random.randint(0, wd)
                    y0 = np.random.randint(0, ht)
                    dx = np.random.randint(bounds[0], bounds[1])
                    dy = np.random.randint(bounds[0], bounds[1])
                    imgs[idx][y0:y0 + dy, x0:x0 + dx, :] = mean_color
        return imgs

    def spatial_transform(self, imgs, flows):
        # randomly sample scale
        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 8) / float(ht),
            (self.crop_size[1] + 8) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = scale
        scale_y = scale
        if np.random.rand() < self.stretch_prob:
            scale_x *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
            scale_y *= 2 ** np.random.uniform(-self.max_stretch, self.max_stretch)
        scale_x = np.clip(scale_x, min_scale, None)
        scale_y = np.clip(scale_y, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            imgs = [cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for img in imgs]

            flows = [cv2.resize(flow, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for flow in flows]
            flows = [flow * [scale_x, scale_y] for flow in flows]

        if self.do_flip:
            if np.random.rand() < self.h_flip_prob:  # h-flip
                imgs = [img[:, ::-1] for img in imgs]
                flows = [flow[:, ::-1] * [-1.0, 1.0] for flow in flows]

            if np.random.rand() < self.v_flip_prob:  # v-flip
                imgs = [img[::-1, :] for img in imgs]
                flows = [flow[::-1, :] * [1.0, -1.0] for flow in flows]

        if imgs[0].shape[0] == self.crop_size[0]:
            y0 = 0
        else:
            y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0])
        if imgs[0].shape[1] == self.crop_size[1]:
            x0 = 0
        else:
            x0 = np.random.randint(0, imgs[0].shape[1] - self.crop_size[1])

        imgs = [img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for img in imgs]
        flows = [flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for flow in flows]

        return imgs, flows

    def __call__(self, imgs, flows):
        # imgs = self.color_transform(imgs)
        # imgs = self.eraser_transform(imgs)
        imgs, flows = self.spatial_transform(imgs, flows)

        imgs = [np.ascontiguousarray(img) for img in imgs]
        flows = [np.ascontiguousarray(flow) for flow in flows]

        return imgs, flows


class SparseFlowAugmentor:
    def __init__(self, crop_size, min_scale=-0.2, max_scale=0.5, do_flip=False):
        # spatial augmentation params
        self.crop_size = crop_size
        self.min_scale = min_scale
        self.max_scale = max_scale
        self.spatial_aug_prob = 0.8
        self.stretch_prob = 0.8
        self.max_stretch = 0.2

        # flip augmentation params
        self.do_flip = do_flip
        self.h_flip_prob = 0.5
        self.v_flip_prob = 0.1

        # photometric augmentation params
        self.photo_aug = ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.3 / 3.14)
        self.asymmetric_color_aug_prob = 0.2
        self.eraser_aug_prob = 0.5

    def color_transform(self, imgs):

        img_num = len(imgs)
        image_stack = np.concatenate(imgs, axis=0)
        image_stack = np.array(self.photo_aug(Image.fromarray(image_stack)), dtype=np.uint8)
        imgs = np.split(image_stack, img_num, axis=0)

        return imgs

    def resize_sparse_flow_map(self, flow, valid, fx=1.0, fy=1.0):
        ht, wd = flow.shape[:2]
        coords = np.meshgrid(np.arange(wd), np.arange(ht))
        coords = np.stack(coords, axis=-1)

        coords = coords.reshape(-1, 2).astype(np.float32)
        flow = flow.reshape(-1, 2).astype(np.float32)
        valid = valid.reshape(-1).astype(np.float32)

        coords0 = coords[valid >= 1]
        flow0 = flow[valid >= 1]

        ht1 = int(round(ht * fy))
        wd1 = int(round(wd * fx))

        coords1 = coords0 * [fx, fy]
        flow1 = flow0 * [fx, fy]

        xx = np.round(coords1[:, 0]).astype(np.int32)
        yy = np.round(coords1[:, 1]).astype(np.int32)

        v = (xx > 0) & (xx < wd1) & (yy > 0) & (yy < ht1)
        xx = xx[v]
        yy = yy[v]
        flow1 = flow1[v]

        flow_img = np.zeros([ht1, wd1, 2], dtype=np.float32)
        valid_img = np.zeros([ht1, wd1], dtype=np.int32)

        flow_img[yy, xx] = flow1
        valid_img[yy, xx] = 1

        return flow_img, valid_img

    def spatial_transform(self, imgs, flows, valids):
        pad_t = 0
        pad_b = 0
        pad_l = 0
        pad_r = 0
        if self.crop_size[0] > imgs[0].shape[0]:
            # pad_t = self.crop_size[0] - img1.shape[0]
            pad_b = self.crop_size[0] - imgs[0].shape[0]
        if self.crop_size[1] > imgs[0].shape[1]:
            print("[In kitti data, padding along width axis now!]")
            pad_r = self.crop_size[1] - imgs[0].shape[1]
        if pad_b != 0 or pad_r != 0 or pad_t != 0:
            imgs = [np.pad(img, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant',
                           constant_values=((0, 0), (0, 0), (0, 0))) for img in imgs]
            flows = [np.pad(flow, ((pad_t, pad_b), (pad_l, pad_r), (0, 0)), 'constant',
                            constant_values=((0, 0), (0, 0), (0, 0))) for flow in flows]
            valids = [np.pad(valid, ((pad_t, pad_b), (pad_l, pad_r)), 'constant', constant_values=((0, 0), (0, 0))) for
                      valid in valids]
        # randomly sample scale

        ht, wd = imgs[0].shape[:2]
        min_scale = np.maximum(
            (self.crop_size[0] + 1) / float(ht),
            (self.crop_size[1] + 1) / float(wd))

        scale = 2 ** np.random.uniform(self.min_scale, self.max_scale)
        scale_x = np.clip(scale, min_scale, None)
        scale_y = np.clip(scale, min_scale, None)

        if np.random.rand() < self.spatial_aug_prob:
            # rescale the images
            imgs = [cv2.resize(img, None, fx=scale_x, fy=scale_y, interpolation=cv2.INTER_LINEAR) for img in imgs]
            for idx in range(len(flows)):
                flows[idx], valids[idx] = self.resize_sparse_flow_map(flows[idx], valids[idx], fx=scale_x, fy=scale_y)

        if self.do_flip:
            if np.random.rand() < 0.5:  # h-flip
                imgs = [img[:, ::-1] for img in imgs]
                flows = [flow[:, ::-1] * [-1.0, 1.0] for flow in flows]
                valids = [valid[:, ::-1] for valid in valids]

        margin_y = 20
        margin_x = 50

        y0 = np.random.randint(0, imgs[0].shape[0] - self.crop_size[0] + margin_y)
        x0 = np.random.randint(-margin_x, imgs[0].shape[1] - self.crop_size[1] + margin_x)

        y0 = np.clip(y0, 0, imgs[0].shape[0] - self.crop_size[0])
        x0 = np.clip(x0, 0, imgs[0].shape[1] - self.crop_size[1])

        imgs = [img[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for img in imgs]

        flows = [flow[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for flow in flows]

        valids = [valid[y0:y0 + self.crop_size[0], x0:x0 + self.crop_size[1]] for valid in valids]

        return imgs, flows, valids

    def __call__(self, imgs, flows, valids):
        imgs = self.color_transform(imgs)
        imgs, flows, valids = self.spatial_transform(imgs, flows, valids)

        imgs = [np.ascontiguousarray(img) for img in imgs]
        flows = [np.ascontiguousarray(flow) for flow in flows]
        valids = [np.ascontiguousarray(valid) for valid in valids]

        return imgs, flows, valids