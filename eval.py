
import os
import torch
import os.path as osp
import numpy as np
import matplotlib
import sys

DISPLAY = 'DISPLAY' in os.environ
if not DISPLAY:
    matplotlib.use('Agg')
import matplotlib.pyplot as plt
from tqdm import tqdm
from tools.options import Options
from network.robustloc import RobustLoc
from torchvision import transforms
from tools.utils import quaternion_angular_error, qexp, load_state_dict
from data.dataloaders import RobotCar
from torch.utils.data import DataLoader
from torch.autograd import Variable
from tools.options import Options
from network import resnet
opt = Options().parse()




def main(epoch='0', 
    t_median_error_best=[10000., 10000., 0], 
    q_median_error_best=[10000., 10000., 0], 
    tq_median_error_best=[10000., 10000., 0]):

    # Config
    opt = Options().parse()
    cuda = torch.cuda.is_available()
    device = "cuda"

    # Model
    feature_extractor = resnet.resnet34(pretrained=False)

    model = RobustLoc(feature_extractor)
    model.eval()


    t_criterion = lambda t_pred, t_gt: np.linalg.norm(t_pred - t_gt)
    q_criterion = quaternion_angular_error

    stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)
    tforms = []
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    data_transform = transforms.Compose(tforms)
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())



    # read mean and stdev for un-normalizing predictions
    pose_stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'pose_stats.txt')
    pose_m, pose_s = np.loadtxt(pose_stats_file)  # mean and stdev

    # Load the dataset
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, train=False, transform=data_transform, 
        target_transform=target_transform, seed=opt.seed, subseq_length=opt.subseq_length)
    data_set = RobotCar(**kwargs)
    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}

    loader = DataLoader(data_set, batch_size=64, shuffle=False, **kwargs)


    pred_poses_list = []
    targ_poses_list = []

    # load weights
    model.to(device)
    weights_filename = f'logs/{opt.dataset}_{opt.scene}/models/epoch_{str(epoch)}.pth.tar'




    if osp.isfile(weights_filename):
        checkpoint = torch.load(weights_filename, map_location=device)
        load_state_dict(model, checkpoint['model_state_dict'])
        print('Loaded weights from {:s}'.format(weights_filename))
    else:
        print('Could not load weights from {:s}'.format(weights_filename))
        sys.exit(-1)



    pbar = tqdm(enumerate(loader))
    for idx, (data, target, image_name_pure, timestamps_list) in pbar:
        
        b, subseq_length, c, h, w = data.shape


        data_var = Variable(data, requires_grad=False)
        data_var = data_var.to(device)
        timestamps_var = timestamps_list.to(device)


        target = target[:,opt.subseq_length//2,:]

        with torch.set_grad_enabled(False):
            output, outputl3, outputl2, outputl1, output_final, output_ae = model(data_var, timestamps_var) # [64,3,256,256]



        output = output_final.view(b, opt.subseq_length, -1)[:, opt.subseq_length//2, :]
        

        s = output.size()
        output = output.cpu().data.numpy().reshape((-1, s[-1]))
        target = target.numpy().reshape((-1, s[-1]))


        # normalize the predicted quaternions
        q = [qexp(p[3:]) for p in output]
        output = np.hstack((output[:, :3], np.asarray(q)))
        q = [qexp(p[3:]) for p in target]
        target = np.hstack((target[:, :3], np.asarray(q)))

        # un-normalize the predicted and target translations
        output[:, :3] = (output[:, :3] * pose_s) + pose_m
        target[:, :3] = (target[:, :3] * pose_s) + pose_m



        for each_output in output:
            pred_poses_list.append(each_output)
        for each_target in target:
            targ_poses_list.append(each_target)


    pred_poses_nanmean = np.vstack(pred_poses_list)
    targ_poses_nanmean = np.vstack(targ_poses_list)
    np.savetxt('output_poses/ep{:s}_pred.txt'.format(str(epoch)), pred_poses_nanmean, fmt='%8.8f')
    np.savetxt('output_poses/ep{:s}_targ.txt'.format(str(epoch)), targ_poses_nanmean, fmt='%8.8f')



    # calculate losses
    t_loss_nanmean = np.asarray([t_criterion(p, t) for p, t in zip(pred_poses_nanmean[:, :3], targ_poses_nanmean[:, :3])])
    q_loss_nanmean = np.asarray([q_criterion(p, t) for p, t in zip(pred_poses_nanmean[:, 3:], targ_poses_nanmean[:, 3:])])

    t_median_error_nanmean = np.median(t_loss_nanmean)
    q_median_error_nanmean = np.median(q_loss_nanmean)
    t_mean_error_nanmean = np.mean(t_loss_nanmean)
    q_mean_error_nanmean = np.mean(q_loss_nanmean)



    if True:
        fig, axs = plt.subplots()
        real_pose = (pred_poses_nanmean[:, :3] - pose_m) / pose_s
        gt_pose = (targ_poses_nanmean[:, :3] - pose_m) / pose_s
        plt.plot(real_pose[:, 1], real_pose[:, 0], color='red', linewidth=0.5)


        # ---- colorful line
        norm = plt.Normalize(t_loss_nanmean.min(), t_loss_nanmean.max())
        norm_y = norm(t_loss_nanmean)
        plt.scatter(gt_pose[:, 1], gt_pose[:, 0], c=norm_y, cmap='jet',linewidths=1)



        plt.xlabel('x [km]')
        plt.ylabel('y [km]')
        plt.plot(gt_pose[0, 1], gt_pose[0, 0], 'y*', markersize=15)
        image_filename = osp.join(osp.expanduser(opt.results_dir), '{:s}.png'.format(str(epoch)))
        fig.savefig(image_filename)
        plt.close()
        

    return (
        t_median_error_nanmean, q_median_error_nanmean, t_mean_error_nanmean, q_mean_error_nanmean
        )
    




if __name__ == '__main__':
    t_median_error_best = [10000., 10000., 0]
    q_median_error_best = [10000., 10000., 0]
    tq_median_error_best = [10000., 10000., 0]
    t_median_error_nanmean, q_median_error_nanmean, t_mean_error_nanmean, q_mean_error_nanmean = main('31', t_median_error_best, q_median_error_best,tq_median_error_best)
    print('t median {:3.2f} m,  mean {:3.2f} m \nq median {:3.2f} degrees, mean {:3.2f} degree'\
            .format(t_median_error_nanmean, t_mean_error_nanmean, q_median_error_nanmean, q_mean_error_nanmean))