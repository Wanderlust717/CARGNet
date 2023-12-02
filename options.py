import argparse
import os
import torch

class CDTrainOptions():
    def __init__(self):
        self.parser = argparse.ArgumentParser()
        self.initialized = False

    def initialize(self):
        self.parser.add_argument('--cuda', action='store_true', default=True)
        self.parser.add_argument('--datadir', type=str, default='/mnt/vdf/jyq/CARGNet/datasets/DSIFN/train',
                                 help='the txt path of the changed images and labels')#'/mnt/vdf/jyq/CARGNet/datasets/AICD/train'
        self.parser.add_argument('--data-test-path', type=str, default='/mnt/vdf/jyq/CARGNet/datasets/DSIFN/test/',
                                 help='the txt path of the testing images and labels')#'/mnt/vdf/jyq/CARGNet/datasets/LEVIR-CD/test/C/'
        self.parser.add_argument('--label-path', type=str,default='/mnt/vdf/jyq/CARGNet/datasets/DSIFN/test/label/',
                                 help='the path of test label ')#'/mnt/vdf/jyq/CARGNet/datasets/AICD/test/test/all/CSCEN/label/'
        self.parser.add_argument('--dataset_name', type=str, default="DSIFN", help='name of the dataset')
        self.parser.add_argument('--batch_size', type=int, default=8, help='size of the batches')
        self.parser.add_argument('--lr', type=float, default=0.001, help='adam: learning rate')
        self.parser.add_argument('--img_height', type=int, default=256, help='size of image height')
        self.parser.add_argument('--img_width', type=int, default=256, help='size of image width')
        self.parser.add_argument('--resume', action='store_true', help="load the pretrained pth from model_result_dir")
        self.parser.add_argument('--model_result_dir', type=str, default='saved_models/DSIFN', help=' where to save the checkpoints')


        #-------------------------------------------------------------------------------------------------#


    def parse(self):
        if not self.initialized:
            self.initialize()
        args = self.parser.parse_args()

        os.makedirs('%s' % ( args.img_result_dir), exist_ok=True)
        os.makedirs('%s' % ( args.model_result_dir), exist_ok=True)

        print('------------ Options -------------')
        with open("%s/args.log" % (args.model_result_dir) ,"w") as args_log:
            for k, v in sorted(vars(args).items()):
                print('%s: %s ' % (str(k), str(v)))
                args_log.write('%s: %s \n' % (str(k), str(v)))

        print('-------------- End ----------------')

        self.args = args
        return self.args