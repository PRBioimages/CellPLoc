import torch
import torch.nn.functional as F
import torch.nn as nn


class ConLoss(nn.Module):
    """Following Supervised Contrastive Learning:
        https://arxiv.org/pdf/2004.11362.pdf."""

    def __init__(self, temperature=0.07, base_temperature=0.07, reduction='mean'):  # 0.07
        super().__init__()
        self.temperature = temperature
        self.base_temperature = base_temperature
        self.reduction = reduction

    def forward(self, features, mask=None, batch_size=-1):
        device = (torch.device('cuda')
                  if features.is_cuda
                  else torch.device('cpu'))

        if mask is not None:
            # SupCon loss (Partial Label Mode)
            mask = mask.float().detach().to(device)
            # compute logits
            anchor_dot_contrast = torch.div(
                torch.matmul(features[:batch_size].to(torch.float32), features.T.to(torch.float32)),
                self.temperature).to(torch.float32)
            # for numerical stability
            logits_max, _ = torch.max(anchor_dot_contrast, dim=1, keepdim=True)
            logits = anchor_dot_contrast - logits_max.detach()

            # mask-out self-contrast cases
            logits_mask = torch.scatter(
                torch.ones_like(mask),
                1,
                torch.arange(batch_size).view(-1, 1).to(device),
                0
            )
            mask = mask * logits_mask

            # compute log_prob
            exp_logits = torch.exp(logits) * logits_mask
            log_prob = logits.to(torch.float32) - torch.log(exp_logits.sum(1, keepdim=True) + 1e-12).to(torch.float32)

            # compute mean of log-likelihood over positive
            mean_log_prob_pos = (mask.to(torch.float32) * log_prob.to(torch.float32)).sum(1) / mask.sum(1)

            # loss
            loss = - (self.temperature / self.base_temperature) * mean_log_prob_pos.to(torch.float32)
            # loss = loss.mean()
        else:
            # MoCo loss (unsupervised)
            # compute logits
            # Einstein sum is more intuitive
            # positive logits: Nx1
            q = features[:batch_size]
            k = features[batch_size:batch_size * 2]
            queue = features[batch_size * 2:]
            l_pos = torch.einsum('nc,nc->n', [q.to(torch.float32), k.to(torch.float32)]).unsqueeze(-1)
            # negative logits: NxK
            l_neg = torch.einsum('nc,kc->nk', [q.to(torch.float32), queue.to(torch.float32)])
            # logits: Nx(1+K)
            logits = torch.cat([l_pos.to(torch.float32), l_neg.to(torch.float32)], dim=1)
            logits = logits.to(torch.float32)

            # apply temperature
            logits /= self.temperature

            # labels: positive key indicators
            labels = torch.zeros(logits.shape[0], dtype=torch.long).cuda()
            loss = F.cross_entropy(logits.to(torch.float32), labels, reduction=self.reduction)

        return loss.float()

