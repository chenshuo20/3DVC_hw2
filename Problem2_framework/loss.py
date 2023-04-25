import torch
from torch import nn
from scipy.spatial.distance import directed_hausdorff

class CDLoss(nn.Module):
    """
    CD Loss.
    """

    def __init__(self):
        super(CDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        # TODO: Implement CD Loss
        # Example:
        #     cd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return cd_loss

        cd_loss = torch.sum(torch.norm(prediction - ground_truth, p=2))
        return cd_loss


class HDLoss(nn.Module):
    """
    HD Loss.
    """
    
    def __init__(self):
        super(HDLoss, self).__init__()
    
    def forward(self, prediction, ground_truth):
        # TODO: Implement HD Loss
        # Example:
        #     hd_loss = torch.tensor(0, dtype=torch.float32, device=prediction.device)
        #     return hd_loss
        # Compute the Euclidean distance map
        dist_map = torch.cdist(prediction, ground_truth, p=2)
        
        hd_dist = torch.max(torch.min(dist_map, dim=1)[0])
        hd_loss = hd_dist
    
        return hd_loss
