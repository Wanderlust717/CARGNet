import torch
from loss import *
from datasets import *
from options import CDTrainOptions
import torch.nn as nn
from test import test
from eval import eval1
from utils import *
from tqdm import tqdm
import time
import torch.nn.functional as F
from torch.autograd import Variable
from sklearn.neighbors import DistanceMetric
import augmentations
from manual_seed import set_seed, adjust_lr
from models.networks import WSCD
import torchvision.transforms as transforms
from PIL import Image
import numpy as np
import os
CUDA_LAUNCH_BLOCKING = 1
os.environ["CUDA_VISIBLE_DEVICES"] = "6"


def data_augmentation(image1, image2, uimage1, uimage2):
    preprocess = transforms.Compose([transforms.Resize((256, 256)),
                                     transforms.ToTensor(),
                                     transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
                                     ])
    aug_list = augmentations.augmentations

    ws = np.float32(
        np.random.dirichlet([1] * 3))
    m = np.float32(np.random.beta(1, 1))
    aug1 = []
    aug2 = []

    B, _, _, _ = image1.shape
    for index in range(B):
        aug = np.random.choice([True, False])
        img1 = Image.fromarray(image1[index].numpy())
        img2 = Image.fromarray(image2[index].numpy())
        # cdm = mask[index].detach().cpu().numpy().transpose(1, 2, 0)

        img1_tensor = preprocess(img1)
        img2_tensor = preprocess(img2)

        if aug:

            mix = torch.zeros_like(img1_tensor)
            mix2 = torch.zeros_like(img2_tensor)

            for i in range(3):  # three paths
                image_aug = img1.copy()
                image_aug2 = img2.copy()

                depth = np.random.randint(1, 4)
                for _ in range(depth):
                    idx = np.random.choice([0, 1])
                    # if idx == 0:
                    op = np.random.choice(aug_list)
                    image_aug = op(image_aug, 1)
                    image_aug2 = op(image_aug2, 1)


                # Preprocessing commutes since all coefficients are convex

                mix += ws[i] * preprocess(image_aug)
                mix2 += ws[i] * preprocess(image_aug2)

            mixed = (1 - m) * img1_tensor + m * mix
            mixed2 = (1 - m) * img2_tensor + m * mix2

            aug1.append(mixed)
            aug2.append(mixed2)

        else:
            aug1.append(img1_tensor)
            aug2.append(img2_tensor)

    cg1, cg2 = torch.stack(aug1).cuda(), torch.stack(aug2).cuda()  # change imag

    return cg1, cg2

def balanced_binary_cross_entropy(predicted, target):
    n_0 = torch.sum(target<0.5).item() # 负样本数量
    n_1 = torch.sum(target>=0.5).item() # 正样本数量
    n = n_0 + n_1 # 样本总数
    w_0 = n / (2 * n_0+1e-6) # 负样本权重
    w_1 = n / (2 * n_1+1e-6) # 正样本权重
    w = torch.tensor([w_0, w_1], dtype=torch.float32).to(predicted.device)
    weight = torch.zeros(target.shape)
    weight[target<0.5]=w_0
    weight[target>=0.5] =w_1
    weight=weight.to(predicted.device)
    bbce = F.binary_cross_entropy(predicted, target, weight=weight)
    return bbce


def cal_protypes(feat, label_data, pred):
    feat = F.interpolate(feat, size=label_data.size()[-2:], mode='bilinear')
    b, c, h, w = feat.size()
    prototypes = torch.zeros((b,1, feat.shape[1]), dtype=feat.dtype,device=feat.device)
    for i in range(b):
        cur_mask=label_data[i]
        cur_mask_=cur_mask.expand(c,h,w).view(c,-1)
        cur_feat = feat[i]
        cur_feat_=cur_feat.view(c,-1)
        cur_prototype = torch.zeros((1, c),dtype=feat.dtype,device=feat.device)
        cls_feat=(cur_feat_[cur_mask_==1]).view(c, -1).sum(-1)/(cur_mask.sum()+1e-6)
        cur_prototype[0,:]=cls_feat
        prototypes[i] += cur_prototype
    return prototypes


def Expand(feat,vecs,pred,label_data):
    b, k, oh, ow = pred.size()
    pred = F.interpolate(pred, size=feat.size()[-2:], mode='bilinear')
    _, _, h, w = pred.size()
    vecs = vecs.view(b, k, -1, 1, 1)
    feat = feat.view(b, 1, -1, h, w)
    d = torch.abs(feat - vecs).mean(2)
    res=torch.exp(-d)
    label_data = F.interpolate(label_data, size=feat.size()[-2:], mode='bilinear')
    res[label_data==1]=1
    pred_idx = np.argwhere(pred.cpu().detach().numpy() < 0.7)
    res[tuple(pred_idx.T)] = 0
    res = F.interpolate(res, size=(oh, ow), mode='bilinear')
    res=res.detach()
    return res

def requires_grad(model, flag=True):
    for p in model.parameters():
        p.requires_grad = flag

def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    set_seed(1)
    #load the args
    args = CDTrainOptions().parse()

    C_train = os.path.join(args.datadir, "C")
    cloder1 = Get_dataloader(C_train, args.batch_size, reshape_size=(args.img_height, args.img_width), model="CD")

    UC_train = os.path.join(args.datadir, "UC")
    dis_UC_loder = Get_dataloader(UC_train, args.batch_size, reshape_size=(args.img_height, args.img_width), model="CD")

    real = iter(cloder1)
    dis_UC = iter(dis_UC_loder)

    # 开始训练
    # Initialize network
    generator=WSCD()
    generator=generator.cuda()
    # Loss functions
    criterion_GAN, criterion_pixelwise = Get_loss_func(args)
    # Optimizers
    optimizer_G = torch.optim.SGD(generator.parameters(),lr=args.lr, momentum=0.9,weight_decay=0.0005)
    log = {'bestfm_iter': 0, 'best_mae': 10, 'fm': 0, 'bestfm_it': 0, 'best_fm': 0, 'mae': 0, 'R': 0, "P": 0}
    log_file = open('%s/train_log_dsifn.txt' % (args.model_result_dir), 'w')
    f = open('%s/best_dsifn.txt' % (args.model_result_dir), 'w')
    tbar = tqdm(range(0, 40000))
    for i in tbar:
        try:
            # --------img1 img2 label name ---------
            img1, img2, img1_numpy, img2_numpy, point_label, name = next(real)
            img1 = img1.to(device)
            img2 = img2.to(device)
            point_label[point_label==0.88235295]=1
            point_label = point_label.to(device)


            dis_UC_img1, dis_UC_img2, dis_UC_img1_numpy, dis_UC_img2_numpy, _, uname = next(dis_UC)
            dis_UC_img1 = dis_UC_img1.to(device)
            dis_UC_img2 = dis_UC_img2.to(device)

        except (OSError, StopIteration):

            # --------img1 img2 label name ---------
            real = iter(cloder1)
            dis_UC = iter(dis_UC_loder)
            img1, img2, img1_numpy, img2_numpy, point_label, name = next(real)
            img1 = img1.to(device)
            img2 = img2.to(device)
            point_label[point_label == 0.88235295] = 1
            point_label = point_label.to(device)

            dis_UC_img1, dis_UC_img2, dis_UC_img1_numpy, dis_UC_img2_numpy, _, uname = next(dis_UC)
            dis_UC_img1 = dis_UC_img1.to(device)
            dis_UC_img2 = dis_UC_img2.to(device)

        plr = adjust_lr(optimizer_G, i, args.lr)
        # ------------------
        #  Train Generators
        # ------------------

        optimizer_G.zero_grad()
        requires_grad(generator, True)


        loss_fn = nn.BCELoss()
        L_con = torch.nn.MSELoss()

        img1_aug, img2_aug = data_augmentation(img1_numpy, img2_numpy, dis_UC_img1_numpy,dis_UC_img2_numpy)
        ocf1_g,oc1_g,ocf2_g,oc2_g,_,ouc1_g,_,ouc2_g,sc1_g,sc2_g= generator([img1_aug, img2_aug], [dis_UC_img1, dis_UC_img2])


        zerol = Variable(torch.zeros_like(ouc1_g).cuda(), requires_grad=False)



        indices = np.argwhere(point_label.cpu().numpy() == 1)
        loss_p = loss_fn(oc1_g[tuple(indices.T)], point_label[tuple(indices.T)])

        prototypes=cal_protypes(ocf2_g,point_label,oc2_g)
        expand_anotations = Expand(ocf2_g,prototypes,oc2_g,point_label)

        loss_e = balanced_binary_cross_entropy(oc2_g,expand_anotations)
        L_con_value = L_con(oc1_g, oc2_g)


        loss_umask1g = criterion_pixelwise(ouc1_g, zerol)
        loss_umask2g = criterion_pixelwise(ouc2_g, zerol)
        loss_umask =loss_umask1g+loss_umask2g


        loss_cmse1g = criterion_pixelwise(sc1_g, oc1_g)
        loss_cmse2g = criterion_pixelwise(sc2_g, oc2_g)
        loss_cmse =loss_cmse1g+loss_cmse2g

        loss_G = loss_p+loss_e+L_con_value+loss_umask + loss_cmse
        loss_G.backward()
        optimizer_G.step()

        localtime = time.asctime(time.localtime(time.time()))
        tbar.set_description('Time: %s | Iter: %d | loss_p: %f | loss_G: %f | loss_e: %f | loss_con: %f '
                             % (localtime, i, loss_p.data.cpu(), loss_G.data.cpu(),loss_e.data.cpu(),L_con_value.data.cpu()))
        log_file.write('Time: %s | Iter: %d | loss_p: %f | loss_G: %f | loss_e: %f | loss_con: %f\n'
                       % (localtime, i, loss_p.data.cpu(), loss_G.data.cpu(),loss_e.data.cpu(),L_con_value.data.cpu()))
        log_file.flush()

        if (i + 1) % 1000 == 0:
            # Save model checkpoints
            torch.save(generator.state_dict(), '%s/generator_latest.pth' % (args.model_result_dir))

            mask_save_path = '%s/test/' % (args.model_result_dir)
            image_path = args.data_test_path
            test(generator, mask_save_path, image_path)
            gt_path = args.label_path
            mae1, fmeasure1, recall, precision,OA,_ = eval1(mask_save_path, gt_path, 2)

            if fmeasure1[1] > log['best_fm']:
                log['bestfm_iter'] = i
                log['best_fm'] = fmeasure1[1]
                log['mae'] = mae1
                log['R'] = recall[1]
                log['P'] = precision[1]
                torch.save(generator.state_dict(), '%s/generator_best.pth' % (args.model_result_dir))

            # print('====================================================================================================================')
            # print('Iter:', i, "mae:", mae1, "fmeasure:", fmeasure1, "R:",recall, "P:",precision)
            # print('bestfm_iter:', log['bestfm_iter'], 'mae:', log['mae'], 'best_fm:', log['best_fm'], "R:", log['R'], "P:", log['P'])
            # print('====================================================================================================================')

            f.write(
                '====================================================================================================================\n')
            f.write("Iter: {}, mae: {}, fmeasure: {}, R: {}, P: {}\n".format(i, mae1, fmeasure1[1], recall[1], precision[1]))
            f.write("bestfm_iter: {}, mae: {}, best_fm: {}, R: {}, P: {}\n".format(log['bestfm_iter'], log['mae'],
                                                                                   log['best_fm'], log['R'], log['P']))
            f.write(
                '====================================================================================================================\n\n')
            f.flush()
    f.close()


if __name__ == "__main__":
    main()
