import torch
import torch.nn as nn


class SoftMatchWeighting(nn.Module):
    """
    SoftMatch learnable truncated Gaussian weighting
    """

    def __init__(self, num_classes, n_sigma=2, momentum=0.999, per_class=False):
        super().__init__()
        self.num_classes = num_classes
        self.n_sigma = n_sigma
        self.per_class = per_class
        self.m = momentum

        # initialize Gaussian mean and variance
        # 应该考虑不同类别的不同情况，而不是一视同仁
        if not self.per_class:
            self.prob_max_mu_t = torch.tensor(1.0 / self.num_classes, dtype=torch.float64)
            self.prob_max_var_t = torch.tensor(1.0, dtype=torch.float64)
        else:
            self.prob_mu_t = torch.ones(self.num_classes, dtype=torch.float64) / self.num_classes
            self.prob_var_t = torch.ones(self.num_classes, dtype=torch.float64)

    @torch.no_grad()
    def update(self, probs_x_ulb, labels):
        # max_probs, max_idx = probs_x_ulb.max(dim=-1)
        if not self.per_class:
            # 初始化为全零和全一的张量
            prob_mu_t = torch.zeros(1)  # 使用一个单独的值来存储全局均值
            prob_var_t = torch.ones(1)  # 使用一个单独的值来存储全局方差
            # 只选择标签为正的样本
            positive_mask = labels > 0
            if torch.any(positive_mask):
                # 使用布尔索引来获取所有正标签的概率
                positive_probs = probs_x_ulb[positive_mask]
                # 计算全局均值和方差
                prob_mu_t = torch.mean(positive_probs)
                prob_var_t = torch.var(positive_probs, unbiased=True)
            # 更新存储的全局均值和方差
            self.prob_mu_t = self.m * self.prob_mu_t + (1 - self.m) * prob_mu_t
            self.prob_var_t = self.m * self.prob_var_t + (1 - self.m) * prob_var_t
        else:
            prob_mu_t = torch.zeros_like(self.prob_mu_t)
            prob_var_t = torch.ones_like(self.prob_var_t)
            for i in range(self.num_classes):
                # Select only the samples where the label for class i is present
                class_mask = labels[:, i] > 0
                if torch.any(class_mask):
                    # Calculate mean for class i using only the relevant samples
                    class_probs = probs_x_ulb[class_mask, i]
                    prob_mu_t[i] = torch.mean(class_probs)
                    # 检查样本数量
                    num_samples = class_probs.numel()
                    # 如果只有一个样本，则将方差设置为0
                    if num_samples == 1:
                        prob_var_t[i] = torch.tensor(0.0, dtype=prob_var_t.dtype)
                    else:
                        # 如果有多个样本，则使用无偏估计计算方差
                        prob_var_t[i] = torch.var(class_probs, unbiased=True)
            # Calculate the presence of each class in the batch
            class_present = torch.sum(labels, dim=0) > 0
            self.prob_mu_t[class_present] = self.m * self.prob_mu_t[class_present] + (1 - self.m) * prob_mu_t[class_present]
            self.prob_var_t[class_present] = self.m * self.prob_var_t[class_present] + (1 - self.m) * prob_var_t[class_present]

    @torch.no_grad()
    def masking(self, logits_x_ulb, labels, sigmoid_x_ulb=True, get_mu=False):
        '''
        整体来说是算样本预测概率在总体预测的截断高斯分布中的值
        '''
        if not self.prob_mu_t.is_cuda:
            self.prob_mu_t = self.prob_mu_t.to(logits_x_ulb.device).double()
        if not self.prob_var_t.is_cuda:
            self.prob_var_t = self.prob_var_t.to(logits_x_ulb.device).double()

        if sigmoid_x_ulb:
            probs_x_ulb = torch.sigmoid(logits_x_ulb.double().detach())
        else:
            # logits is already probs
            probs_x_ulb = logits_x_ulb.double().detach()

        self.update(probs_x_ulb, labels)
        mask = torch.ones_like(probs_x_ulb, dtype=torch.float64)
        # compute weight
        if not self.per_class:
            mu = self.prob_mu_t
            var = self.prob_var_t
            # 创建一个布尔掩码，标记probs_x_ulb中不为0的元素
            non_zero_mask = probs_x_ulb != 0
            mask[non_zero_mask] = torch.exp(
                -(
                        (torch.clamp(probs_x_ulb[non_zero_mask] - mu, max=0.0) ** 2)
                        /
                        (2 * var / (self.n_sigma ** 2))
                )
            )
        else:
            for i in range(self.num_classes):
                # 使用每个类的均值和方差
                mu = self.prob_mu_t[i]
                var = self.prob_var_t[i]
                # 初始化掩码值为1，表示默认不应用高斯函数
                mask[:, i] = 1.0
                # 仅对非零的probs_x_ulb元素应用高斯函数
                non_zero_mask = probs_x_ulb[:, i] != 0
                if torch.any(non_zero_mask):
                    # 计算截断高斯函数的值，并应用掩码以仅更新对应类的非零概率
                    mask[non_zero_mask, i] = torch.exp(
                        -(
                                (torch.clamp(probs_x_ulb[non_zero_mask, i] - mu, max=0.0) ** 2)
                                /
                                (2 * var / (self.n_sigma ** 2))
                        )
                    )
        if get_mu:
            prob_mu_t = self.prob_mu_t
            return mask, prob_mu_t
        else:
            return mask

    @torch.no_grad()
    def get_lbl(self, logits_x_ulb):
        if not self.prob_mu_t.is_cuda:
            self.prob_mu_t = self.prob_mu_t.to(logits_x_ulb.device).double()
        probs_x_ulb = logits_x_ulb.double().detach()
        cell_lbls = probs_x_ulb
        for i in range(self.num_classes):
            # 使用每个类的均值
            mu = self.prob_mu_t[i]
            lbl_filtered_0 = probs_x_ulb[:, i] - mu < 0
            lbl_filtered_1 = probs_x_ulb[:, i] - mu >= 0
            if torch.any(lbl_filtered_0):
                cell_lbls[lbl_filtered_0, i] = 0
            if torch.any(lbl_filtered_1):
                cell_lbls[lbl_filtered_1, i] = 1
        return cell_lbls

