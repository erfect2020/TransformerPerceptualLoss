import torch
import torch.nn.functional as F
import torch.cuda.amp as amp

EPS=1e-8


def projectedDistributionLoss(x, y, num_projections=1000):
    '''Projected Distribution Loss (https://arxiv.org/abs/2012.09289)
    x.shape = B,M,N,...
    '''
    def rand_projections(dim, device=torch.device('cpu'), num_projections=1000):
        projections = torch.randn((dim,num_projections), device=device)
        projections = projections / torch.sqrt(torch.sum(projections ** 2, dim=0, keepdim=True))    # columns are unit length normalized
        return projections
    e_x = x
    e_y = y
    loss = 0
    for ii in range(e_x.shape[1]):
#        g = torch.sort(e_x[:,:,ii],dim=1)[0] - torch.sort(e_y[:,:,ii],dim=1)[0]; print(g.mean(), g.min(), g.max())
        loss = loss + F.mse_loss(torch.sort(e_x[:,ii,:],dim=1)[0] , torch.sort(e_y[:,ii,:],dim=1)[0] , reduction='mean') ** 0.5  # if this gives issues; try Huber loss later
    return loss


