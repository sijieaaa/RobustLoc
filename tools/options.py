import argparse
import os






class Options():
    def __init__(self):
        self.parser = argparse.ArgumentParser(formatter_class=argparse.ArgumentDefaultsHelpFormatter)

    def initialize(self):

        # e.g 
        # RobotCar's dir is /home/sijie/3090tmp_home/sijie/Documents/loc/RobotCar/
        self.parser.add_argument('--data_dir', type=str, default='/home/sijie/3090tmp_home/sijie/Documents/loc/')



        self.parser.add_argument('--cuda', type=float, default=0)
        self.parser.add_argument('--nThreads', type=int, default=6)
        self.parser.add_argument('--resume_epoch', type=int, default=-1)
        self.parser.add_argument('--dataset', type=str, default='RobotCar')
        self.parser.add_argument('--scene', type=str, default='loop') # loop, full
        self.parser.add_argument('--subseq_length', type=int, default=5) # 5 for loop, 7 for full
        self.parser.add_argument('--skip', type=int, default=10)

        self.parser.add_argument('--odefc', type=int, default=1)
        self.parser.add_argument('--gattnorm', type=str, default='ln')
        self.parser.add_argument('--gattactivation', type=str, default='relu')



        self.parser.add_argument('--cropsize', type=int, default=128)
        self.parser.add_argument('--random_crop', type=bool, default=True)
        self.parser.add_argument('--color_jitter', type=float, default=1)
        

        self.parser.add_argument('--batchsize', type=int, default=64)
        self.parser.add_argument('--seed', type=int, default=7)
        self.parser.add_argument('--logdir', type=str, default='./logs')
        self.parser.add_argument('--exp_name', type=str, default='name')


        self.parser.add_argument('--epochs', type=int, default=300)
        self.parser.add_argument('--beta', type=float, default=-3.0)
        self.parser.add_argument('--gamma', type=float, default=-3.0)
        self.parser.add_argument('--results_dir', type=str, default='figures')
        self.parser.add_argument('--models_dir', type=str, default='models')
        self.parser.add_argument('--lr', type=float, default=2e-4)
        self.parser.add_argument('--weight_decay', type=float, default=0.0005)

        



    def parse(self):
        self.initialize()
        self.opt = self.parser.parse_args()


        # save to the disk
        self.opt.exp_name = f'{self.opt.dataset}_{self.opt.scene}'
        expr_dir = os.path.join(self.opt.logdir, self.opt.exp_name)
        self.opt.results_dir = os.path.join(expr_dir, self.opt.results_dir)
        self.opt.models_dir = os.path.join(expr_dir, self.opt.models_dir)
        mkdirs([self.opt.logdir, expr_dir, self.opt.models_dir, self.opt.results_dir])

        
        os.environ['CUDA_VISIBLE_DEVICES'] = str(self.opt.cuda)

        return self.opt


def mkdirs(paths):
    if isinstance(paths, list) and not isinstance(paths, str):
        for path in paths:
            if not os.path.exists(path):
                os.mkdir(path)
    else:
        if not os.path.exists(path):
            os.mkdir(paths)
