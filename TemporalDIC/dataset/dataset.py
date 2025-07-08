import torch
import torch.utils.data as data
import argparse
import os
import random

from utils.frame_utils import *

import pandas as pd


class FlowDataset(data.Dataset):
    def __init__(self, aug_params=None, sparse=False, oneside=False, input_frames=5, reverse_rate=0.3):
        self.augmentor = None
        self.sparse = sparse
        self.oneside = oneside
        self.input_frames = input_frames
        print("[input frame number is {}]".format(self.input_frames))
        self.reverse_rate = reverse_rate
        print("[reverse_rate is {}]".format(self.reverse_rate))
        self.augmentor = FlowAugmentor(**aug_params)

        self.init_seed = False
        self.flow_list = []
        self.image_list = []
        self.has_gt_list = []

    def __getitem__(self, index):

        if not self.init_seed:
            worker_info = torch.utils.data.get_worker_info()
            if worker_info is not None:
                # print(worker_info.id)
                torch.manual_seed(worker_info.id)
                np.random.seed(worker_info.id)
                random.seed(worker_info.id)
                self.init_seed = True

        index = index % len(self.image_list)

        valids = None

        if self.oneside and not self.sparse:
            flows = []
            np.seterr(divide='ignore', invalid='ignore')
            for idx, name in enumerate(
                    self.flow_list[index][::2]):
                dis_x = pd.read_csv(self.flow_list[index][2 * idx], header=None).values
                dis_y = pd.read_csv(self.flow_list[index][2 * idx + 1], header=None).values
                dis_x = np.expand_dims(dis_x, axis=2)
                dis_y = np.expand_dims(dis_y, axis=2)
                flow = np.concatenate((dis_x, dis_y), axis=2)
                flows.append(flow)

        elif self.oneside and self.sparse:
            flows = []
            valids = []
            for idx in range(len(self.has_gt_list[index])):  
                if self.has_gt_list[index][idx]:  
                    flow, valid = readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow)
                    valids.append(valid)
                else:  
                    flow, valid = readFlowKITTI(self.flow_list[index][idx])
                    flows.append(flow)
                    valids.append(valid * 0.0)
        else:
            flows = [read_gen(path) for path in self.flow_list[index]] 


        flows_new = []
        flows_new.append(flows[2])
        flows_new.append(flows[3])
        flows_new.append(flows[4])


        imgs = [read_gen(path) for path in self.image_list[index]]

        flows = [np.array(flow).astype(np.float32) for flow in
                 flows_new]

        imgs = [np.array(img).astype(np.uint8) for img in
                imgs]
        if len(imgs[0].shape) == 2:
            imgs = [np.expand_dims(img, axis=2) for img in imgs]
        else:
            imgs = [img[..., :3] for img in imgs]

        imgs = [torch.from_numpy(img).permute(2, 0, 1).float() for img in
                imgs]
        flows = [torch.from_numpy(flow).permute(2, 0, 1).float() for flow in flows]

        if valids is None:
            valids = [((flow[0].abs() < 1000) & (flow[1].abs() < 1000)).float() for flow in flows]
        else:
            valids = [torch.from_numpy(valid).float() for valid in valids]

        return torch.stack(imgs), torch.stack(flows), torch.stack(valids)

    def __rmul__(self, v):  
        self.flow_list = v * self.flow_list
        self.image_list = v * self.image_list
        self.has_gt_list = v * self.has_gt_list
        return self

    def __len__(self):
        return len(self.image_list)


class SP(FlowDataset):
    def __init__(self, aug_params=None, input_frames=5,
                 root='',
                 annotations_path='',
                 train=True):
        super(SP, self).__init__(aug_params=aug_params, input_frames=input_frames, oneside=True, sparse=False)

        self.image_list = []
        self.flow_list = []
        self.has_gt_list = []

        annotations = pd.read_csv(annotations_path, header=None)
        Ref_list = annotations.iloc[:, 0].tolist()
        Def_list = annotations.iloc[:, 1].tolist()
        DisX_list = annotations.iloc[:, 2].tolist()
        DisY_list = annotations.iloc[:, 3].tolist()
        # print(Def_list)
        for i, name in enumerate(Def_list):
            Def_list[i] = name.replace("Def", "Frame")

        for i, name in enumerate(Def_list[::5]):  
            index = Def_list.index(name)

            Def_path = os.path.join(root, "img")
            self.image_list.append(
                [os.path.join(Def_path, Def_list[index]), os.path.join(Def_path, Def_list[index + 1]),
                 os.path.join(Def_path, Def_list[index + 2]), os.path.join(Def_path, Def_list[index + 3]),
                 os.path.join(Def_path, Def_list[index + 4])])

            Dis_path = os.path.join(root, "dis")
            self.flow_list.append(
                [os.path.join(Dis_path, DisX_list[index]), os.path.join(Dis_path, DisY_list[index]),
                 os.path.join(Dis_path, DisX_list[index + 1]), os.path.join(Dis_path, DisY_list[index + 1]),
                 os.path.join(Dis_path, DisX_list[index + 2]), os.path.join(Dis_path, DisY_list[index + 2]),
                 os.path.join(Dis_path, DisX_list[index + 3]), os.path.join(Dis_path, DisY_list[index + 3]),
                 os.path.join(Dis_path, DisX_list[index + 4]), os.path.join(Dis_path, DisY_list[index + 4])])
            self.has_gt_list.append([True] * (input_frames - 2) + [True] * (input_frames - 2))
def fetch_dataloader(args, TRAIN_DS='S'):
    aug_params = {'crop_size': args.image_size, 'min_scale': -0.2, 'max_scale': 0.6, 'do_flip': True}
    SP_instance = SP(aug_params, root=r'./data',
                     annotations_path=r"./data/Annotations.csv", train=False)


    train_loader = data.DataLoader(SP_instance, batch_size=args.batch_size,
                                   pin_memory=False, shuffle=True, num_workers=args.batch_size * 2, drop_last=True)

    print('Training with %d image pairs' % len(SP_instance))
    return train_loader
