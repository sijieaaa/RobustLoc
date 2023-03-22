import os



import torch
import sys
import time
import os.path as osp
import numpy as np


from tqdm import tqdm
from tools.options import Options
from network.robustloc import RobustLoc
from torchvision import transforms
from tools.utils import AtLocPlusCriterion
from data.dataloaders import RobotCar
from torch.utils.data import DataLoader
from torch.autograd import Variable
import eval
import cv2
from network import resnet
from tools.utils import mkdir_custom
opt = Options().parse()
from tools.utils import set_seed
from tools.utils import load_state_dict
set_seed(7)
os.environ["CUDA_VISIBLE_DEVICES"] = str(opt.cuda)



def main():


    # Config

    cuda = torch.cuda.is_available()
    device = "cuda"

    # Model
    feature_extractor = resnet.resnet34(pretrained=True)


    model = RobustLoc(feature_extractor)
    param_list = [{'params': model.parameters()}]
    kwargs = dict(saq=opt.beta, srq=opt.gamma, learn_beta=True, learn_gamma=True)
    train_criterion = AtLocPlusCriterion(**kwargs)


    # Optimizer
    param_list = [{'params': model.parameters()}]
    if hasattr(train_criterion, 'sax') and hasattr(train_criterion, 'saq'):
        print('learn_beta')
        param_list.append({'params': [train_criterion.sax, train_criterion.saq]})
    if opt.gamma is not None and hasattr(train_criterion, 'srx') and hasattr(train_criterion, 'srq'):
        print('learn_gamma')
        param_list.append({'params': [train_criterion.srx, train_criterion.srq]})
    optimizer = torch.optim.Adam(param_list, lr=opt.lr, weight_decay=opt.weight_decay)


    stats_file = osp.join(opt.data_dir, opt.dataset, opt.scene, 'stats.txt')
    stats = np.loadtxt(stats_file)



    tforms = []
    if opt.color_jitter > 0:
        assert opt.color_jitter <= 1.0
        print('Using ColorJitter data augmentation')
        tforms.append(transforms.ColorJitter(brightness=opt.color_jitter, contrast=opt.color_jitter, saturation=opt.color_jitter, hue=0.5))
    else:
        print('Not Using ColorJitter')
    tforms.append(transforms.ToTensor())
    tforms.append(transforms.Normalize(mean=stats[0], std=np.sqrt(stats[1])))
    data_transform = transforms.Compose(tforms)
    target_transform = transforms.Lambda(lambda x: torch.from_numpy(x).float())



    # Load the dataset
    kwargs = dict(scene=opt.scene, data_path=opt.data_dir, transform=data_transform, 
        target_transform=target_transform, seed=opt.seed, subseq_length=opt.subseq_length)


    train_set = RobotCar(train=True, **kwargs)


    kwargs = {'num_workers': opt.nThreads, 'pin_memory': True} if cuda else {}
    train_loader = DataLoader(train_set, batch_size=opt.batchsize, shuffle=True, **kwargs)



    # ---- resume from epoch
    if opt.resume_epoch>0:
        weights_filename = f'logs/{opt.dataset}_{opt.scene}/models/epoch_{str(epoch)}.pth.tar'

        if osp.isfile(weights_filename):
            checkpoint = torch.load(weights_filename, map_location=device)
            load_state_dict(model, checkpoint['model_state_dict'])
            print('Resume weights from {:s}'.format(weights_filename))
        else:
            print('Could not load weights from {:s}'.format(weights_filename))
            sys.exit(-1)



    model.to(device)
    train_criterion.to(device)
    experiment_name = opt.exp_name

    t_median_error_best_nanmean = [10000., 10000., 0]
    q_median_error_best_nanmean = [10000., 10000., 0]
    tq_median_error_best_nanmean = [10000., 10000., 0, 10000.]
    tq_mean_error_best_nanmean = [10000., 10000., 0, 10000.]



    t0 = time.time()
    for epoch in range(opt.resume_epoch+1, opt.epochs):
        txt = []



        model.train()





        pbar = tqdm(enumerate(train_loader))
        for batch_idx, (data, target, timestamps_list) in pbar:

            b, subseq_length, c, h, w = data.shape


            # # ---- numpy save
            # if opt.save_images:
            #     tmp_data = data[0,0,:3].cpu().permute(1,2,0).numpy()*255
            #     cv2.imwrite('output_images/ep0_batch{:d}_image.png'.format(int(batch_idx)), tmp_data)



            
            # # ---- cat depth
            data_var = Variable(data, requires_grad=True) # [16,10,3,256,256]
            target_var = Variable(target, requires_grad=False) # [16,10,6]



            data_var = data_var.to(device)
            target_var = target_var.to(device)
            timestamps_var = timestamps_list.to(device)

            target_var = target_var.view(-1,6)


            with torch.set_grad_enabled(True):
                output, outputl3, outputl2, outputl1, output_final, output_ae = model(data_var, timestamps_var)

                


                loss_tmp = train_criterion(output, target_var)
                loss_tmp += train_criterion(outputl3, target_var)
                loss_tmp += train_criterion(output_final, target_var)





            loss_tmp.backward()
            optimizer.step()
            optimizer.zero_grad()
            now_lr = optimizer.param_groups[0]["lr"]



        # ---- save
        filename = osp.join(opt.models_dir, 'epoch_{:s}.pth.tar'.format(str(epoch)))
        checkpoint_dict = {'epoch': epoch, 'model_state_dict': model.state_dict(), 'optim_state_dict': optimizer.state_dict(), 'criterion_state_dict': train_criterion.state_dict()}
        torch.save(checkpoint_dict, filename)



        # ---- test
        t_median_error_nanmean, q_median_error_nanmean, t_mean_error_nanmean, q_mean_error_nanmean = eval.main(
            str(epoch), t_median_error_best_nanmean, q_median_error_best_nanmean, tq_median_error_best_nanmean)
        txt.append('t median {:3.2f} m,  mean {:3.2f} m \nq median {:3.2f} degrees, mean {:3.2f} degree'\
            .format(t_median_error_nanmean, t_mean_error_nanmean, q_median_error_nanmean, q_mean_error_nanmean))

        if t_median_error_nanmean < t_median_error_best_nanmean[0]:
            t_median_error_best_nanmean = [t_median_error_nanmean, q_median_error_nanmean, epoch]
        if q_median_error_nanmean < q_median_error_best_nanmean[1]:
            q_median_error_best_nanmean = [t_median_error_nanmean, q_median_error_nanmean, epoch]
        if t_median_error_nanmean+q_median_error_nanmean<tq_median_error_best_nanmean[-1]:
            tq_median_error_best_nanmean = [t_median_error_nanmean, q_median_error_nanmean, epoch, t_median_error_nanmean+q_median_error_nanmean]
        if t_mean_error_nanmean+q_mean_error_nanmean<tq_mean_error_best_nanmean[-1]:
            tq_mean_error_best_nanmean = [t_mean_error_nanmean, q_mean_error_nanmean, epoch, t_mean_error_nanmean+q_mean_error_nanmean]



        txt.append('t  median best {:.2f}/{:.2f}\t Epoch {:d}'.format(*t_median_error_best_nanmean))
        txt.append('q  median best {:.2f}/{:.2f}\t Epoch {:d}'.format(*q_median_error_best_nanmean))
        txt.append('tq median best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_median_error_best_nanmean))
        txt.append('tq mean   best {:.2f}/{:.2f}\t Epoch {:d}'.format(*tq_mean_error_best_nanmean))




        # ---- print   
        txt.append('Train/test {:s}\t Epoch {:d}\t Lr {:f}\t Time {:.2f}'.format(
            experiment_name, epoch, now_lr, time.time()-t0))
        txt.append('-----------------------')


        # ---- print and save txt
        f = open('results.txt', 'a')
        for info in txt:
            print(info)
            f.write(info)
            f.write('\n')
        f.close()

        t0 = time.time()





if __name__ == '__main__':
    set_seed(7)


    mkdir_custom('output_images/')
    mkdir_custom('output_poses/')

    if osp.exists('results.txt'):
        os.remove('results.txt')
    

    main()