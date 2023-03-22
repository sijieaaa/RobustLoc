import os
import random
import torch
import numpy as np
import os.path as osp
from tools.utils import process_poses, load_image
from torch.utils import data
from tools.options import Options
from torchvision import transforms
import torchvision.transforms.functional as TVF
from tools.utils import set_seed
set_seed(7)
opt = Options().parse()




class RobotCar(data.Dataset):
    def __init__(self, scene, data_path, train, transform=None, target_transform=None, real=False, 
        skip_images=False, seed=7, undistort=False, vo_lib='stereo', subseq_length=10):

        np.random.seed(seed)
        self.transform = transform
        self.target_transform = target_transform
        self.skip_images = skip_images
        self.undistort = undistort
        self.subseq_length = subseq_length

        self.train = train


        data_dir = osp.join(data_path, 'RobotCar', scene)


        # ---- train test split
        if scene=='loop': # for loop (within day)
            if train:
                seqs = [
                    # '2014-06-23-15-36-04',
                    '2014-06-23-15-41-25',
                    # '2014-06-26-08-53-56',
                    '2014-06-26-09-24-58', 
                ]
            elif not train:
                seqs = [
                    '2014-06-23-15-36-04',
                    # '2014-06-23-15-41-25',
                    '2014-06-26-08-53-56',
                    # '2014-06-26-09-24-58', 
                ]


        # if scene=='loop': # for loop (cross day)
        #     if train:
        #         seqs = [
        #             # '2014-06-23-15-36-04',
        #             # '2014-06-23-15-41-25',
        #             '2014-06-26-08-53-56',
        #             '2014-06-26-09-24-58', 
        #         ]
        #     elif not train:
        #         seqs = [
        #             '2014-06-23-15-36-04',
        #             '2014-06-23-15-41-25',
        #             # '2014-06-26-08-53-56',
        #             # '2014-06-26-09-24-58', 
        #         ]



        elif scene=='full':
            if train:
                seqs = [
                    '2014-11-28-12-07-13',
                    '2014-12-02-15-30-08',
                ]
            elif not train:
                seqs = [
                    '2014-12-09-13-21-02',
                ]

            

        ps = {}
        ts = {}
        self.img_paths = []
        self.depth_paths = []

        for seq in seqs:
            seq_dir = osp.join(data_dir, seq)

            if self.train:
                ts[seq] = [int(image_name.split('.png')[0]) for image_name in os.listdir(osp.join(seq_dir, 'stereo', 'centre_128'))] 
            elif not self.train:
                ts[seq] = [int(image_name.split('.png')[0]) for image_name in os.listdir(osp.join(seq_dir, 'stereo', 'centre_128'))] 



            ts[seq] = sorted(ts[seq]) 
            if seq=='2014-06-23-15-41-25':
                ts[seq] = ts[seq][:-13]
                assert len(ts[seq]) == 3343






            
            if self.train:
                self.img_paths.extend(
                    [osp.join(seq_dir, 'stereo', 'centre_{:s}'.format(str(opt.cropsize)), '{:d}.png'.format(t)) for t in ts[seq]])


            elif not self.train:
                self.img_paths.extend(
                    [osp.join(seq_dir, 'stereo', 'centre_{:s}'.format(str(opt.cropsize)), '{:d}.png'.format(t)) for t in ts[seq]])





        pose_stats_filename = osp.join(data_dir, 'pose_stats.txt')
        mean_t, std_t = np.loadtxt(pose_stats_filename)

        if self.train: # train
            if scene=='loop':
                ps = {
                    '2014-06-23-15-36-04':np.loadtxt('data/RobotCar_poses/2014-06-23-15-36-04_tR.txt'),
                    '2014-06-23-15-41-25':np.loadtxt('data/RobotCar_poses/2014-06-23-15-41-25_tR.txt'),
                    '2014-06-26-08-53-56':np.loadtxt('data/RobotCar_poses/2014-06-26-08-53-56_tR.txt'),
                    '2014-06-26-09-24-58':np.loadtxt('data/RobotCar_poses/2014-06-26-09-24-58_tR.txt'), 
                }
            elif scene=='full':
                ps = {
                    '2014-11-28-12-07-13':np.loadtxt('data/RobotCar_poses/2014-11-28-12-07-13_tR.txt'),
                    '2014-12-02-15-30-08':np.loadtxt('data/RobotCar_poses/2014-12-02-15-30-08_tR.txt'),
                    '2014-12-09-13-21-02':np.loadtxt('data/RobotCar_poses/2014-12-09-13-21-02_tR.txt'),
                }


        elif not self.train: # test
            if scene=='loop':
                ps = {
                    '2014-06-23-15-36-04':np.loadtxt('data/RobotCar_poses/2014-06-23-15-36-04_tR.txt'),
                    '2014-06-23-15-41-25':np.loadtxt('data/RobotCar_poses/2014-06-23-15-41-25_tR.txt'),
                    '2014-06-26-08-53-56':np.loadtxt('data/RobotCar_poses/2014-06-26-08-53-56_tR.txt'),
                    '2014-06-26-09-24-58':np.loadtxt('data/RobotCar_poses/2014-06-26-09-24-58_tR.txt'), 
                }
            elif scene=='full':
                ps = {
                    '2014-11-28-12-07-13':np.loadtxt('data/RobotCar_poses/2014-11-28-12-07-13_tR.txt'),
                    '2014-12-02-15-30-08':np.loadtxt('data/RobotCar_poses/2014-12-02-15-30-08_tR.txt'),
                    '2014-12-09-13-21-02':np.loadtxt('data/RobotCar_poses/2014-12-09-13-21-02_tR.txt'),
                }





        self.samples_length = []
        self.poses = np.empty((0, 6))
        for seq in seqs:
            pss = process_poses(poses_in=ps[seq], mean_t=mean_t, std_t=std_t,
                              align_R=np.eye(3), align_t=np.zeros(3),
                              align_s=1)
            self.poses = np.vstack((self.poses, pss))
            self.samples_length.append(len(pss))



    
    def __len__(self):
        return len(self.poses)
        


    def get_indices(self, index): 
        if self.train:
            if index >= self.samples_length[0]:
                indices_pool = np.array(range(self.samples_length[0], self.__len__()))
            else:
                indices_pool = np.array(range(0, self.samples_length[0]))
        
        elif not self.train:
            if index >= self.samples_length[0]:
                indices_pool = np.array(range(self.samples_length[0], self.__len__()))
            else:
                indices_pool = np.array(range(0, self.samples_length[0]))


        output_indices = np.arange(-(opt.subseq_length//2)*opt.skip, (opt.subseq_length//2)*opt.skip+1, opt.skip) + index



        # ---- check if outrange
        for i_index, each_index in enumerate(output_indices):
            if each_index > indices_pool[-1]:
                output_indices[i_index] = indices_pool[-1]
            elif each_index < indices_pool[0]:
                output_indices[i_index] = indices_pool[0]

        return output_indices
        

    def __getitem__(self, index):


        image_name = self.img_paths[index].split('/')[-1]
        image_name_pure = image_name.split('.')[0]
        

        indices_list = self.get_indices(index)
        

        timestamps_list = []
        for each_index in indices_list:
            image_name = self.img_paths[each_index].split('/')[-1]
            image_name_pure = image_name.split('.')[0]
            each_timestamp = int(image_name_pure)
            each_timestamp = torch.tensor(each_timestamp)
            timestamps_list.append(each_timestamp)
        timestamps_list = torch.stack(timestamps_list, dim=0)



        if self.train:
            imgs = []

            for index in indices_list:
                img = load_image(self.img_paths[index])



                # ---- crop
                if opt.random_crop:
                    i, j, th, tw = transforms.RandomCrop(size=opt.cropsize).get_params(
                        img, output_size=[opt.cropsize, opt.cropsize])
                    img = TVF.crop(img, i, j, th, tw)
                else:
                    img = transforms.CenterCrop(opt.cropsize)(img)
                

                
                imgs.append(img)
            


        elif not self.train:
            imgs = []
            for index in indices_list:
                img = load_image(self.img_paths[index])
                img = transforms.CenterCrop(opt.cropsize)(img)



                imgs.append(img)


        
        poses = []
        for index in indices_list:
            pose = np.float32(self.poses[index])
            poses.append(pose)




        for index_in_subseq, _ in enumerate(poses):
            if self.target_transform is not None:
                poses[index_in_subseq] = self.target_transform(poses[index_in_subseq])
            if self.transform is not None:
                imgs[index_in_subseq] = self.transform(imgs[index_in_subseq])




        # ---- return
        if self.train:
            imgs = torch.stack(imgs)
            poses = torch.stack(poses)
            return imgs, poses, timestamps_list

        else:
            imgs = torch.stack(imgs)
            poses = torch.stack(poses)
            return imgs, poses, image_name_pure, timestamps_list


