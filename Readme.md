# Transformer Perceptual Loss for Image Deblurring
----------
This repository is an official implementation of **Image Deblurring by Exploring In-depth Properties of Transformer**.


# Basic usage

```
from loss.deblur_loss import ReconstructPerceptualLoss as ReconstructLoss



model = yourmodel()
criterion = ReconstructLoss(opt)
model = model.cuda()
criterion.pretrain_mae = criterion.pretrain_mae.to(torch.device('cuda'))

for index, train_data in tqdm(enumerate(train_loader)):
        gt, b_img = train_data
        b_img = b_img.cuda()
        gt_img = gt.cuda()
        x = b_img
        recover_img = model(x)
        losses = criterion(recover_img, gt_img)
        grad_loss = losses["total_loss"]
        optimizer.zero_grad()
        grad_loss.backward()
        optimizer.step()
```

If this repo help you, please cite us:
```
@article{liang2024image,
  title={Image deblurring by exploring in-depth properties of transformer},
  author={Liang, Pengwei and Jiang, Junjun and Liu, Xianming and Ma, Jiayi},
  journal={IEEE Transactions on Neural Networks and Learning Systems},
  year={2024},
  publisher={IEEE}
}
```
