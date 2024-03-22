import torch
import torch.nn as nn
from torch.nn import MSELoss, L1Loss
from loss.InfoNCE import InfoNCE
import einops
from loss.generate_index import compute_index_mat


class contrastiveLoss(nn.Module):
    def __init__(self):
        super(contrastiveLoss, self).__init__()
        self.infonce = InfoNCE()

    def forward(self, query, positive_key, negative_keys):
        losses = {}
        query = einops.rearrange(query, 'b c p -> (b c) p')
        positive_key = einops.rearrange(positive_key, 'b c p -> (b c) p')
        negative_keys = einops.rearrange(negative_keys, 'b n c p -> (b c n) p')
        output = self.infonce(query, positive_key, negative_keys)
        losses["total_loss"] = output
        return losses

def chunk_cosine_sim(x: torch.Tensor, y: torch.Tensor) -> torch.Tensor:
    """ Computes cosine similarity between all possible pairs in two sets of vectors.
    Operates on chunks so no large amount of GPU RAM is required.
    :param x: an tensor of descriptors of shape Bx1x(t_x)xd' where d' is the dimensionality of the descriptors and t_x
    is the number of tokens in x.
    :param y: a tensor of descriptors of shape Bx1x(t_y)xd' where d' is the dimensionality of the descriptors and t_y
    is the number of tokens in y.
    :return: cosine similarity between all descriptors in x and all descriptors in y. Has shape of Bx1x(t_x)x(t_y) """
    result_list = []
    num_token_x = x.shape[2]
    for token_idx in range(num_token_x):
        token = x[:, :, token_idx, :].unsqueeze(dim=2)  # Bx1x1xd'
        result_list.append(torch.nn.CosineSimilarity(dim=3)(token, y))  # Bx1xt
    return torch.stack(result_list, dim=2)  # Bx1x(t_x)x(t_y)



class selfPerceptualLoss(nn.Module):
    def __init__(self, img_size):
        super(selfPerceptualLoss, self).__init__()
        self.l2_loss = MSELoss()
        self.l1_loss = L1Loss(reduction='mean')

    def forward(self, query, positive_key, ids):
        output = 0
        for q, p in zip(query, positive_key):
            output += self.l1_loss(q, p)
        return output
