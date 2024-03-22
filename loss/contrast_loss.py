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



class selfContrastiveLoss(nn.Module):
    def __init__(self, img_size):
        super(selfContrastiveLoss, self).__init__()
        self.infonce = InfoNCE(negative_mode='paired')
        self.l2_loss = MSELoss()
        self.l1_loss = L1Loss(reduction='mean')
        # self.mini_index = compute_index_mat(img_size // 16, img_size // 16, 1024)
        # self.L = 256

    def forward(self, query, positive_key, ids):
        # output = self.l2_loss(query, positive_key)
        output = 0
        for q, p in zip(query, positive_key):
            output += self.l1_loss(q, p)
        # output = self.l1_loss(query, positive_key)

        # query_map = chunk_cosine_sim(query.unsqueeze(1), query.unsqueeze(1))
        # positive_key_map = chunk_cosine_sim(positive_key.unsqueeze(1), positive_key.unsqueeze(1))
        # output = self.l2_loss(query_map ** 3, positive_key_map ** 3)

        # output = self.infonce(query, positive_key, negative_keys)

        return output

        # losses = {}
        # B, C, P = query.shape
        # n_numsneg = 64
        # negative_keys = query.unsqueeze(2).repeat(1, 1, n_numsneg, 1)
        # # arg_sort = torch.argsort(torch.eye(C), dim=1)[:,:C-1]
        # # arg_sort = arg_sort[:, torch.randperm(C-1)][:, :n_numsneg]
        # arg_sort = self.mini_index.clone().to(ids.device)
        # # generate the binary mask: 0 is keep, 1 is remove
        # ids, _ = ids.sort()
        # arg_sort = torch.gather(arg_sort.unsqueeze(0).repeat(B, 1, 1), dim=1, index= ids.unsqueeze(2).repeat(1,1,self.L))
        # full_rank = torch.argsort(arg_sort)
        # filter_rank = torch.gather(full_rank, dim=2, index=ids.unsqueeze(1).repeat(1,C,1))
        # filter_rank, _ = filter_rank.sort(dim=2)
        # unnorm_sort = torch.gather(arg_sort, dim=2, index=filter_rank)
        # norm_value = torch.tensor([torch.arange(0,C).tolist() for _ in range(B)]).unsqueeze(1).repeat(1,C,1).to(ids.device)
        # norm_index = unnorm_sort.argsort(dim=2)
        # norm_sort = torch.gather(norm_value, dim=2, index=norm_index.argsort(dim=2))
        # arg_sort = norm_sort[:,:,1:n_numsneg+1]
        # arg_sort = arg_sort.unsqueeze(3).repeat(1,1,1,P)
        #
        # # arg_sort = arg_sort.unsqueeze(2).unsqueeze(0).repeat(B, 1, 1, P)
        # arg_sort = arg_sort.to(negative_keys.device)
        # negative_keys = torch.gather(negative_keys, dim=1, index=arg_sort)
        # query = einops.rearrange(query, 'b c p -> (b c) p')
        # positive_key = einops.rearrange(positive_key, 'b c p -> (b c) p')
        # # mask = torch.eye(C).unsqueeze(2).unsqueeze(0).repeat(B,1,1,P).bool()
        # # negative_keys = negative_keys.masked_select(~mask).reshape(B,C,C-1,P)
        # negative_keys = einops.rearrange(negative_keys, 'b c n p -> (b c) n p')
        # # negative_keys = einops.rearrange(negative_keys, 'b n c p -> (b c n) p')
        #
        # # output = self.l2_loss(query, positive_key) / (self.l2_loss(query.unsqueeze(1), negative_keys) + 1e-6 +
        # #                                               self.l2_loss(query, positive_key))
        # output = self.l2_loss(query, positive_key)
        # # output = self.infonce(query, positive_key, negative_keys)
        #
        # return output

   # def infonce(self, batch_representations, temperature, batch_size):
   #      joint_mod_loss_sum = 0
   #      for mod in range(len(batch_representations) - 1):
   #          # Negative pairs: everything that is not in the current joint-modality pair
   #          out_joint_mod = torch.cat(
   #              [batch_representations[-1], batch_representations[mod]], dim=0
   #          )
   #          # [2*B, 2*B]
   #          sim_matrix_joint_mod = torch.exp(
   #              torch.mm(out_joint_mod, out_joint_mod.t().contiguous()) / temperature
   #          )
   #          # Mask for remove diagonal that give trivial similarity, [2*B, 2*B]
   #          mask_joint_mod = (
   #              torch.ones_like(sim_matrix_joint_mod)
   #              - torch.eye(2 * batch_size, device=sim_matrix_joint_mod.device)
   #          ).bool()
   #          # Remove 2*B diagonals and reshape to [2*B, 2*B-1]
   #          sim_matrix_joint_mod = sim_matrix_joint_mod.masked_select(
   #              mask_joint_mod
   #          ).view(2 * batch_size, -1)
   #
   #          # Positive pairs: cosine loss joint-modality
   #          pos_sim_joint_mod = torch.exp(
   #              torch.sum(
   #                  batch_representations[-1] * batch_representations[mod], dim=-1
   #              )
   #              / temperature
   #          )
   #          # [2*B]
   #          pos_sim_joint_mod = torch.cat([pos_sim_joint_mod, pos_sim_joint_mod], dim=0)
   #          loss_joint_mod = -torch.log(
   #              pos_sim_joint_mod / sim_matrix_joint_mod.sum(dim=-1)
   #          )
   #          joint_mod_loss_sum += loss_joint_mod
   #
   #      loss = torch.mean(joint_mod_loss_sum)
   #      tqdm_dict = {"loss": loss}
   #      return loss, tqdm_dict