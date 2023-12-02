import argparse
import os
import torch
from tqdm import tqdm
from utils import to_image_test
from torch.autograd import Variable
from datasets import Get_dataloader_test
from models.networks import WSCD
from eval import eval1
device = torch.device("cuda:0")




def test(model,mask_save_path,image_path):
    model.eval()
    dataloder = Get_dataloader_test(image_path, 1, reshape_size=(256,256), model="SGCD")
    with torch.no_grad():
        for i, (img1, img2,img1_numpy,img2_numpy, label, name) in tqdm(enumerate(dataloder)):
            if not torch.cuda.is_available():
                img1 = Variable(img1)
                img2 = Variable(img2)
            else:
                img1 = Variable(img1).cuda()
                img2 = Variable(img2).cuda()
            feat,mask = model(C=[img1, img2], UC=None,test=True)
            os.makedirs(mask_save_path, exist_ok=True)
            to_image_test(mask, path=mask_save_path, name=str(name[0].replace(".jpg", ".png")))


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='PyTorch Template')
    parser.add_argument('--stict', default='/mnt/vdf/jyq/CARGNet/saved_models/LEVIR-CD/generator_best.pth', type=str)
    parser.add_argument('--image_path', default='/mnt/vdf/jyq/CARGNet/datasets/LEVIR-CD/test', type=str)#/mnt/vdf/jyq/CARGNet/datasets/LEVIR-CD/test/C
    parser.add_argument('--mask_save_path', default='./LEVIRCD/jyq_CARGNet/', type=str)
    parser.add_argument('--label-path', type=str,default='/mnt/vdf/jyq/CARGNet/datasets/LEVIR-CD/test1/label/',
                             help='the path of test label ')
    args=parser.parse_args()

    generator2 = WSCD().cuda()
    generator2.load_state_dict(torch.load(args.stict))
    generator2.eval()
    test(generator2,args.mask_save_path,args.image_path)
    gt_path = args.label_path
    mae1, fmeasure1, recall, precision,OA,IOU = eval1(args.mask_save_path, gt_path, 2)
    print("mae:", mae1, "fmeasure:", fmeasure1, "R:", recall, "P:", precision,"OA:",OA,"IOU:",IOU)


























