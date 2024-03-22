import os.path

import torch
import torch.nn as nn
from torchvision.transforms import RandomCrop, Resize
from torch.nn import MSELoss, L1Loss
# from models.models_mae import mae_vit_base_patch16
from models.dual_model_mae import mae_vit_base_patch16
# from models.models_vit import vit_base_patch16_224
# from models.dual_model_mae_kqv import mae_vit_base_patch16
from utils.pos_embed import interpolate_pos_embed
from loss.contrast_loss import selfContrastiveLoss
from torchvision.transforms.functional import normalize
from loss.hist_loss import HistLoss
from loss.pdl import projectedDistributionLoss

from loss.vgg_loss import PerceptualVGGLoss
from utils.pos_embed import interpolate_pos_embed, interpolate_pos_encoding


class ReconstructLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructLoss, self).__init__()
        self.l1_loss = L1Loss(reduction='mean')

        img_size = opt['image_size']
        pretrained_ckpt = opt['pretrain_mae']
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(self.pretrain_mae, checkpoint_model)
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfContrastiveLoss(img_size)


    def forward(self, recover_img, gt):
        losses = {}
        loss_l1 = self.l1_loss(recover_img, gt)
        with torch.no_grad():
            predict_embed, _ = self.pretrain_mae(recover_img)
            gt_embed, _ = self.pretrain_mae(gt)
        contrast_loss = self.constrastive_loss(predict_embed, gt_embed)
        contrast_loss = contrast_loss * 0.1
        losses["l1"] = loss_l1
        losses["constrastive"] = contrast_loss
        losses["total_loss"] = loss_l1 + contrast_loss

        return losses



class ReconstructConstrastiveLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructConstrastiveLoss, self).__init__()
        self.l1_loss = L1Loss(reduction='mean')
        self.mse_loss = MSELoss(reduction='mean')
        self.hist_loss = HistLoss()

        img_size = opt['image_size']
        pretrained_ckpt = opt['pretrain_mae']
        # self.randomcrop = RandomCrop(224)
        self.pretrain_mae = mae_vit_base_patch16(img_size=img_size)
        pretrained_ckpt = os.path.expanduser(pretrained_ckpt)
        checkpoint = torch.load(pretrained_ckpt, map_location='cpu')
        print("Load pre-trained checkpoint from: %s" % pretrained_ckpt)
        checkpoint_model = checkpoint['model']
        state_dict = self.pretrain_mae.state_dict()
        for k in ['head.weight', 'head.bias']:
            if k in checkpoint_model and checkpoint_model[k].shape != state_dict[k].shape:
                print(f"Removing key {k} from pretrained checkpoint")
                del checkpoint_model[k]
        # interpolate position embedding
        interpolate_pos_embed(self.pretrain_mae, checkpoint_model)
        self.pretrain_mae.load_state_dict(checkpoint_model, strict=False)
        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfContrastiveLoss(img_size)
        # self.constrastive_loss = projectedDistributionLoss()
        # self.constrastive_loss = PerceptualLoss({'4': 1., '9': 1., '13': 1., '18': 1.})
        # self.constrastive_loss = PerceptualVGGLoss()

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]

    def forward(self, recover_img, gt):
        losses = {}
        loss_l1 = self.l1_loss(recover_img, gt)
        # loss_l2 = self.mse_loss(recover_img, gt)
        # loss_charbonnier = torch.mean(torch.sqrt(((recover_img - gt) ** 2) + (1e-3 ** 2)))
        # loss_color = self.hist_loss(recover_img, gt)
        # loss_color = 0.5 * loss_color

        # combine_img = self.randomcrop(torch.cat((recover_img, gt), dim=1))
        # recover_img, gt = combine_img[:, :3, :,:], combine_img[:, 3:, :, :]

        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt = normalize(gt, self.normalize_mean, self.normalize_std)
        predict_embed, gt_embed, ids = self.pretrain_mae(recover_img, gt, 0.50)
        # print("predict embed", len(predict_embed), predict_embed[0].shape)
        # kqv_length = predict_embed[0].shape[2]
        # predict_embed = predict_embed[0][:,:,kqv_length - kqv_length//3 : ]
        # gt_embed = gt_embed[0][:,:,kqv_length - kqv_length//3 :]
        # contrast_loss = self.constrastive_loss(predict_embed, gt_embed, ids) * 0.1
        # # contrast_loss = 30 * contrast_loss
        #
        # contrast_loss = contrast_loss
        # self.l1_loss(predict_embed, gt_embed)
        contrast_loss =0
        for predict_e, gt_e in zip(predict_embed, gt_embed):
            contrast_loss += projectedDistributionLoss(predict_e, gt_e) * 1e-4
        # contrast_loss = projectedDistributionLoss(predict_embed, gt_embed) * 5e-6 #NAFNet
        # contrast_loss = self.constrastive_loss(recover_img, gt) * 1e-4
        # contrast_loss = self.constrastive_loss(recover_img, gt) * 1e-4
        losses["l1"] = loss_l1
        losses["constrastive"] = contrast_loss
        # losses["l2"] = loss_l2
        # losses['charbonnier'] = loss_charbonnier
        # losses["color"] = loss_color
        losses["total_loss"] = contrast_loss + loss_l1 #+ loss_color
        # losses['total_loss'] = loss_l2 + contrast_loss
        # losses['total_loss'] = loss_charbonnier + contrast_loss
        # losses['total_loss'] = loss_l1

        return losses


class ReconstructVitConstrastiveLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructVitConstrastiveLoss, self).__init__()
        self.l1_loss = L1Loss(reduction='mean')
        self.mse_loss = MSELoss(reduction='mean')
        self.hist_loss = HistLoss()

        self.pretrain_mae = vit_base_patch16_224(pretrained=True)
        img_size = opt['image_size']
        npatch = (img_size // 16) ** 2
        dim = 768
        patch_size = 16
        stride_hw = [16, 16]
        w = img_size
        h = img_size
        pos_embend = interpolate_pos_encoding(npatch, dim, self.pretrain_mae, patch_size, stride_hw, w, h)
        self.pretrain_mae.pos_embed = nn.Parameter(pos_embend, requires_grad=False)
        self.pretrain_mae.patch_embed.img_size = [img_size, img_size]

        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfContrastiveLoss(img_size)

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]


    def forward(self, recover_img, gt):
        losses = {}
        # loss_l1 = self.l1_loss(recover_img, gt)
        loss_l2 = self.mse_loss(recover_img, gt)
        # loss_color = self.hist_loss(recover_img, gt)
        # loss_color = 0.5 * loss_color

        # combine_img = self.resize(torch.cat((recover_img, gt), dim=1))
        # recover_img, gt = combine_img[:, :3, :,:], combine_img[:, 3:, :, :]
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt = normalize(gt, self.normalize_mean, self.normalize_std)
        # with torch.no_grad():
        predict_embed = self.pretrain_mae(recover_img)
        gt_embed = self.pretrain_mae(gt)
        # print("prdict", predict_embed.shape, gt_embed.shape)
        contrast_loss = self.constrastive_loss(predict_embed, gt_embed, None)
        contrast_loss =  0.003 * contrast_loss
        # contrast_loss = contrast_loss
        # self.l1_loss(predict_embed, gt_embed)
        losses["l2"] = loss_l2
        losses["constrastive"] = contrast_loss
        # losses["color"] = loss_color
        losses["total_loss"] = contrast_loss + loss_l2 #+ loss_color
        # losses['total_loss'] = loss_l2

        return losses

class ReconstructDINOConstrastiveLoss(nn.Module):
    def __init__(self, opt):
        super(ReconstructDINOConstrastiveLoss, self).__init__()
        self.l1_loss = L1Loss(reduction='mean')
        self.mse_loss = MSELoss(reduction='mean')
        self.hist_loss = HistLoss()

        self.pretrain_mae = vit_base_patch16_224(pretrained=True)

        model_type_dict = {
            'vit_small_patch16_224': 'dino_vits16',
            'vit_small_patch8_224': 'dino_vits8',
            'vit_base_patch16_224': 'dino_vitb16',
            'vit_base_patch8_224': 'dino_vitb8'
        }
        model = torch.hub.load('facebookresearch/dino:main', model_type_dict['vit_base_patch16_224'])
        temp_state_dict = model.state_dict()
        temp_state_dict['head.weight'] = self.pretrain_mae.state_dict()['head.weight']
        temp_state_dict['head.bias'] = self.pretrain_mae.state_dict()['head.bias']
        # del temp_state_dict['head.weight']
        # del temp_state_dict['head.bias']
        self.pretrain_mae.load_state_dict(temp_state_dict)

        img_size = opt['image_size']
        npatch = (img_size // 16) ** 2
        dim = 768
        patch_size = 16
        stride_hw = [16, 16]
        w = img_size
        h = img_size
        pos_embend = interpolate_pos_encoding(npatch, dim, self.pretrain_mae, patch_size, stride_hw, w, h)
        self.pretrain_mae.pos_embed = nn.Parameter(pos_embend, requires_grad=False)
        self.pretrain_mae.patch_embed.img_size = [img_size, img_size]

        for _, p in self.pretrain_mae.named_parameters():
            p.requires_grad = False
        self.constrastive_loss = selfContrastiveLoss(img_size)

        self.normalize_mean = [0.485, 0.456, 0.406]
        self.normalize_std = [0.229, 0.224, 0.225]


    def forward(self, recover_img, gt):
        losses = {}
        # loss_l1 = self.l1_loss(recover_img, gt)
        loss_l2 = self.mse_loss(recover_img, gt)
        # loss_color = self.hist_loss(recover_img, gt)
        # loss_color = 0.5 * loss_color

        # combine_img = self.resize(torch.cat((recover_img, gt), dim=1))
        # recover_img, gt = combine_img[:, :3, :,:], combine_img[:, 3:, :, :]
        recover_img = normalize(recover_img, self.normalize_mean, self.normalize_std)
        gt = normalize(gt, self.normalize_mean, self.normalize_std)
        # with torch.no_grad():
        predict_embed = self.pretrain_mae(recover_img)
        gt_embed = self.pretrain_mae(gt)
        # print("prdict", predict_embed.shape, gt_embed.shape)
        contrast_loss = self.constrastive_loss(predict_embed, gt_embed, None)
        contrast_loss = 0.01 * contrast_loss
        # contrast_loss = contrast_loss
        # self.l1_loss(predict_embed, gt_embed)
        losses["l2"] = loss_l2
        losses["constrastive"] = contrast_loss
        # losses["color"] = loss_color
        losses["total_loss"] = contrast_loss + loss_l2 #+ loss_color
        # losses['total_loss'] = loss_l1

        return losses
