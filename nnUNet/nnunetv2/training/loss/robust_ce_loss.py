import torch
from torch import nn, Tensor
import numpy as np
import torch.nn.functional as F


class RobustCrossEntropyLoss(nn.CrossEntropyLoss):
    """
    this is just a compatibility layer because my target tensor is float and has an extra dimension

    input must be logits, not probabilities!
    """
    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        return super().forward(input, target.long())
    

# class RobustFocalCrossEntropyLoss(nn.CrossEntropyLoss):
#     """
#     this is just a compatibility layer because my target tensor is float and has an extra dimension

#     input must be logits, not probabilities!
#     """
#     def forward(self, input: Tensor, target: Tensor) -> Tensor:
#         if target.ndim == input.ndim:
#             assert target.shape[1] == 1
#             target = target[:, 0]
#         return super().forward(input, target.long())
    
class RobustFocalLoss(nn.Module):
    """
    RobustCrossEntropyLoss와 동일한 호환성 처리(float target + 여분 채널 차원)를
    유지한 focal loss.

    input must be logits, not probabilities!

    FL(p_t) = -alpha_t * (1 - p_t)**gamma * log(p_t)
    gamma=0, alpha=None 이면 일반 cross entropy와 동일.
    """
    def __init__(self, gamma: float = 2.0, alpha=None,
                 ignore_index: int = -100, reduction: str = 'mean'):
        super().__init__()
        self.gamma = gamma
        self.ignore_index = ignore_index
        self.reduction = reduction
        # alpha: 스칼라(전경 가중치) 또는 (C,) per-class 텐서
        if alpha is not None and not isinstance(alpha, Tensor):
            alpha = torch.as_tensor(alpha, dtype=torch.float32)
        self.register_buffer('alpha', alpha)

    def forward(self, input: Tensor, target: Tensor) -> Tensor:
        # --- 원본과 동일한 robust target 처리 ---
        if target.ndim == input.ndim:
            assert target.shape[1] == 1
            target = target[:, 0]
        target = target.long()

        # --- log_softmax + gather 로 정확한 p_t 계산 ---
        valid = target != self.ignore_index
        target_c = torch.where(valid, target, torch.zeros_like(target))

        logpt = F.log_softmax(input, dim=1)
        logpt = logpt.gather(1, target_c.unsqueeze(1)).squeeze(1)  # (N, *spatial)
        pt = logpt.exp()

        loss = -((1.0 - pt) ** self.gamma) * logpt  # focal CE (per-pixel)

        if self.alpha is not None:
            at = self.alpha if self.alpha.ndim == 0 \
                 else self.alpha.to(input.device)[target_c]
            loss = at * loss

        loss = loss * valid  # ignore_index 위치 제거

        if self.reduction == 'mean':
            return loss.sum() / valid.sum().clamp_min(1)
        elif self.reduction == 'sum':
            return loss.sum()
        return loss  # reduction='none'


class TopKLoss(RobustCrossEntropyLoss):
    """
    input must be logits, not probabilities!
    """
    def __init__(self, weight=None, ignore_index: int = -100, k: float = 10, label_smoothing: float = 0):
        self.k = k
        super(TopKLoss, self).__init__(weight, False, ignore_index, reduce=False, label_smoothing=label_smoothing)

    def forward(self, inp, target):
        target = target[:, 0].long()
        res = super(TopKLoss, self).forward(inp, target)
        num_voxels = np.prod(res.shape, dtype=np.int64)
        res, _ = torch.topk(res.view((-1, )), int(num_voxels * self.k / 100), sorted=False)
        return res.mean()
