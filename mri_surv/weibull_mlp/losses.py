import torch
import torch.nn as nn

class WeibullLoss(nn.Module):

    def __init__(self):
        super(WeibullLoss, self).__init__()

    def __call__(self, pred: torch.Tensor, labels: torch.Tensor) -> torch.Tensor:
        """Defines Weibull loss function from XXX et al.

        Args:
            pred (torch.Tensor): 2D tensor with dimensions (num_subjects, num_timepoints)
            labels (torch.Tensor): 2D tensor with dimensions (num_subjects, num_timepoints)

        Returns:
            torch.Tensor: error
        """
        # I'm assuming the predictions and true label tensors will be 2D (num_subjects, num_timepoints)
        error = torch.sub(pred, labels).square()

        # ignore everything after censoring, keeping pre-censored datapoints
        error[labels == -1] = 0
        error = error.sum(dim=1).sum()

        return error
