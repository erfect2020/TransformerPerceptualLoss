import os
import logging
from datetime import datetime
import numpy as np
import random
import torch
import math
from kornia.losses import ssim
# from skimage import measure
from skimage import metrics
import lpips
import torch.distributed as dist


def mkdir(path):
    if not os.path.exists(path):
        os.makedirs(path)


def mkdirs(paths):
    if isinstance(paths, str):
        mkdir(paths)
    else:
        for path in paths:
            mkdir(path)


def get_timestamp():
    return datetime.now().strftime('%y%m%d-%H%M%S')


def mkdir_and_rename(path):
    if os.path.exists(path):
        new_name = path + '_archived_' + get_timestamp()
        print('Path already exists. Rename it to [{:s}]'.format(new_name))
        logger = logging.getLogger('base')
        logger.info('Path already exists. Rename it to [{:s}]'.format(new_name))
        os.rename(path, new_name)
    os.makedirs(path)


def calculate_psnr(img1, img2):
    # img1 and img2 have range [0, 255]
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    img1 = img1.astype(np.float64)
    img2 = img2.astype(np.float64)
    mse = np.mean((img1 - img2)**2)
    if mse == 0:
        return float('inf')
    return 20 * math.log10(1.0 / math.sqrt(mse))


def calculate_ssim(img1, img2):
    # ssim_value = ssim(img1, img2, 11, 'mean')
    # return 1 - ssim_value.item()
    img1 = img1.squeeze().permute(1, 2, 0).cpu().numpy()
    img2 = img2.squeeze().permute(1, 2, 0).cpu().numpy()
    # ssim_value = measure.compare_ssim(img1, img2, data_range=1, multichannel=True)
    ssim_value = metrics.structural_similarity(img1, img2, data_range=1, multichannel=True)
    return ssim_value




def calculate_mae(img1, img2):
    mae = torch.mean((img1 - img2).abs(), dim=[2, 3, 1])
    return mae.squeeze().item()

def calculate_lpips(img1, img2):
    # img1 = lpips.im2tensor(img1)
    # img2 = lpips.im2tensor(img2)
    img1 = img1 * 2.0 - 1
    img2 = img2 * 2.0 - 1
    spatial = False
    loss_fn_vgg = lpips.LPIPS(net='squeeze',spatial=spatial)
    return loss_fn_vgg.forward(img1, img2).squeeze().item()


def setup_logger(logger_name, root, phase, level=logging.INFO, screen=False, tofile=False):
    '''set up logger'''
    lg = logging.getLogger(logger_name)
    formatter = logging.Formatter('%(asctime)s.%(msecs)03d - %(levelname)s: %(message)s',
                                  datefmt='%y-%m-%d %H:%M:%S')
    lg.setLevel(level)
    if tofile:
        log_file = os.path.join(root, phase + '_{}.log'.format(get_timestamp()))
        fh = logging.FileHandler(log_file, mode='w')
        fh.setFormatter(formatter)
        lg.addHandler(fh)
    if screen:
        sh = logging.StreamHandler()
        sh.setFormatter(formatter)
        lg.addHandler(sh)


def set_random_seed(seed):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.random.manual_seed(seed)


def squeeze2d(input, factor):
    if factor == 1:
        return input

    B, C, H, W = input.size()

    assert H % factor == 0 and W % factor == 0, "H or W modulo factor is not 0"

    x = input.view(B, C, H // factor, factor, W // factor, factor)
    x = x.permute(0, 1, 3, 5, 2, 4).contiguous()
    x = x.view(B, C * factor * factor, H // factor, W // factor)

    return x


def setup_for_distributed(is_master):
    """
    This function disables printing when not in master process
    """
    import builtins as __builtin__
    builtin_print = __builtin__.print

    def print(*args, **kwargs):
        force = kwargs.pop('force', False)
        if is_master or force:
            builtin_print(*args, **kwargs)

    __builtin__.print = print


def is_dist_avail_and_initialized():
    if not dist.is_available():
        return False
    if not dist.is_initialized():
        return False
    return True


def get_world_size():
    if not is_dist_avail_and_initialized():
        return 1
    return dist.get_world_size()


def get_rank():
    if not is_dist_avail_and_initialized():
        return 0
    return dist.get_rank()


def init_distributed_mode(opt):
    if opt['train']['dist_on_itp']:
        opt['rank'] = int(os.environ['OMPI_COMM_WORLD_RANK'])
        opt['world_size'] = int(os.environ['OMPI_COMM_WORLD_SIZE'])
        opt['gpu'] = int(os.environ['OMPI_COMM_WORLD_LOCAL_RANK'])
        opt['dist_url'] = "tcp://%s:%s" % (os.environ['MASTER_ADDR'], os.environ['MASTER_PORT'])
        os.environ['LOCAL_RANK'] = str(opt['gpu'])
        os.environ['RANK'] = str(opt['rank'])
        os.environ['WORLD_SIZE'] = str(opt['world_size'])
    elif 'RANK' in os.environ and 'WORLD_SIZE' in os.environ:
        opt['rank'] = int(os.environ["RANK"])
        opt['world_size'] = int(os.environ['WORLD_SIZE'])
        opt['gpu'] = int(os.environ['LOCAL_RANK'])
    else:
        print('Not using distributed mode')
        opt['distributed'] = False
        return

    opt['distributed'] = True

    torch.cuda.set_device(opt['gpu'])
    opt['dist_backend'] = 'nccl'
    print('| distributed init (rank {}): {}, gpu {}'.format(
        opt['rank'], opt['dist_url'], opt['gpu']), flush=True)
    torch.distributed.init_process_group(backend=opt['dist_backend'], init_method=opt['dist_url'],
                                         world_size=opt['world_size'], rank=opt['rank'])
    torch.distributed.barrier()
    setup_for_distributed(opt['rank'] == 0)