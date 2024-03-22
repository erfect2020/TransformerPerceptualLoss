import torch
import torch.nn as nn
from torch.nn import L1Loss
from kornia.filters import  gaussian_blur2d
import torch.nn.functional as F
from kornia.losses import TotalVariation
import time

def generate_vectorc(radius):
    abs_radius = radius.abs()
    basic_disk = torch.ones(abs_radius * 2 + 1, abs_radius * 2 + 1)
    for i in range(abs_radius * 2 + 1):
        for j in range(abs_radius * 2 + 1):
            if ((i-abs_radius) ** 2 + (j-abs_radius) ** 2) > abs_radius ** 2:
                basic_disk[i][j] = 0.0
    sign_radius = radius.sign()
    blur_kernel = torch.zeros_like(basic_disk)
    for i in range(2 * abs_radius):
        center_point = i * sign_radius + abs_radius
        start_point = max(center_point - abs_radius, 0)
        end_point = min(center_point + abs_radius + 1, abs_radius * 2+1)
        blur_kernel[:,start_point:end_point] += basic_disk[:, start_point:end_point] ** 2

    sum_kernel = blur_kernel.sum()
    blur_kernel = blur_kernel/(sum_kernel)
    return blur_kernel


class GemoLoss(nn.Module):
    def __init__(self):
        super(GemoLoss, self).__init__()
        self.mse_loss = L1Loss()
        # MSELoss()
        self.alpha = {
            "unsurpervise": 1000,
            "tv_loss": 1e-5,
        }
        self.beta = 0.1
        self.gamma = 1
        self.radius = 25
        self.epoch = 0
        self.tv_loss = TotalVariation()
        self.radius_set, self.weight_pos_set, self.weight_neg_set = self.radius_dict(self.radius)

    def radius_dict(self, c_radius):
        radius_set = [torch.tensor([[1.]])]
        for i in range(1,c_radius):
            radius_set.append(generate_vectorc(torch.tensor(i)))

        weight_pos_set = []
        weight_neg_set = []
        for i in range(1, c_radius+1):
            current_conv = nn.Conv2d(1, 1, kernel_size=i * 2 + 1, bias=False)
            current_conv.weight.data = radius_set[i-1].unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(False)
            weight_pos_set.append(current_conv.cuda())
        for i in range(1, c_radius + 1):
            current_conv = nn.Conv2d(1, 1, kernel_size=i * 2 + 1, bias=False)
            current_conv.weight.data = radius_set[i-1].flip(1).unsqueeze(0).unsqueeze(0)
            current_conv.requires_grad_(False)
            weight_neg_set.append(current_conv.cuda())

        return radius_set, weight_pos_set, weight_neg_set

    def compute_gemo_none_boost(self, blur_map, x):
        left_gemo = torch.zeros_like(blur_map)
        right_gemo = torch.zeros_like(blur_map)
        b, c, h, w = blur_map.size()
        for i in range(b):
            for j in range(c):
                for k in range(h):
                    for m in range(w):
                        key = blur_map[i,j,k,m]
                        key_int = key.abs().floor().int()
                        left_kernel = self.radius_set[key_int].cuda()
                        right_kernel = self.radius_set[key_int].flip(1).cuda()
                        if key.sign() < 0:
                            left_kernel, right_kernel = right_kernel, left_kernel
                        padding_left = (key_int - k) if k - key_int < 0 else 0
                        padding_right = (k + key_int - h + 1) if k+key_int >= h else 0
                        padding_up = (key_int - m) if m - key_int < 0 else 0
                        padding_down = (m + key_int - w + 1) if m+key_int >= w else 0
                        #if max(padding_up,padding_down,padding_left,padding_right) > 0:
                        #    print('i j k m key_int',i,j,k,m, key_int)
                        local_x = x[i,j,max(0, k - key_int):min(h, k+key_int+1), max(0, m - key_int):min(w, m+key_int+1)]
                        #if max(padding_up,padding_down,padding_left,padding_right) > 0:
                        #    print('local x shape',local_x.shape, padding_left,padding_right,padding_up,padding_down)
                        local_x = F.pad(local_x, (padding_up,padding_down,padding_left,padding_right))
                        #if max(padding_up,padding_down,padding_left,padding_right) > 0:
                        #    print('padding x shape', local_x.shape)
                        left_gemo[i,j,k,m] = (local_x * left_kernel).sum()
                        right_gemo[i,j,k,m] = (local_x * right_kernel).sum()
                        
        
    def compute_gemo(self, blur_map, x):
        left_gemo = torch.zeros_like(blur_map)
        right_gemo = torch.zeros_like(blur_map)

        for i in range(self.radius):
            current_left = F.pad(x[:, :3, :, :], pad=(i, i, i, i))
            current_right = F.pad(x[:, 3:, :, :], pad=(i, i, i, i))

            lpos_mask = ((i - 1 < blur_map) & (blur_map <= i)).float()
            rpos_mask = ((i < blur_map) & (blur_map < i + 1)).float()
            lneg_mask = ((-(i + 1) < blur_map) & (blur_map <= -i)).float()
            rneg_mask = ((-i < blur_map) & (blur_map < -(i - 1))).float()

            pos_mask = (blur_map - i + 1) * lpos_mask + (i + 1 - blur_map) * rpos_mask
            neg_mask = (blur_map + i + 1) * lneg_mask + (-blur_map - i + 1) * rneg_mask

            if i == 0:
                pos_mask = pos_mask * 0.5
                neg_mask = neg_mask * 0.5

            if (pos_mask.sum() + neg_mask.sum()) > 0:
                for j in range(3):
                    left_gemo[:, j:j+1, :, :] += \
                        self.weight_pos_set[i](current_left[:, j:j+1, :, :]) * pos_mask[:, j:j+1, :, :]
                    left_gemo[:, j:j+1, :, :] += \
                        self.weight_neg_set[i](current_left[:, j:j+1, :, :]) * neg_mask[:, j:j+1, :, :]

                    right_gemo[:, j:j+1, :, :] += \
                        self.weight_neg_set[i](current_right[:, j:j+1, :, :]) * pos_mask[:, j:j+1, :, :]
                    right_gemo[:, j:j+1, :, :] += \
                        self.weight_pos_set[i](current_right[:, j:j+1, :, :]) * neg_mask[:, j:j+1, :, :]
        return left_gemo, right_gemo

    def blur_mean_loss(self, img1, img2):
        diff = (img1 - img2)
        diff = gaussian_blur2d(diff, (3, 3), (1.5, 1.5)).abs()
        return diff.mean()

    def forward(self, blur_map, x):
        losses = {}
        
        torch.cuda.synchronize()
        start_time = time.time()
        for _ in range(100):
            left_gemo, right_gemo = self.compute_gemo(blur_map, x)
        end_time = time.time()
        torch.cuda.synchronize()
        print("1. 100X esptime", end_time-start_time)
        
        torch.cuda.synchronize()
        start_time = time.time()
        self.compute_gemo_none_boost(blur_map, x)
        end_time = time.time()
        torch.cuda.synchronize()
        print("2. 1X esptime", end_time-start_time)
        
        unsurpervise_loss = self.blur_mean_loss(left_gemo, right_gemo) * self.alpha["unsurpervise"]
        losses["unsurpervise loss"] = unsurpervise_loss
        tv_loss = self.tv_loss(blur_map).sum() * self.alpha["tv_loss"]
        losses["tv loss"] = tv_loss
        if self.epoch < 3000:
            losses["total_loss"] = unsurpervise_loss
        else:
            losses["total_loss"] = unsurpervise_loss + tv_loss
        self.epoch += 1
        return losses
