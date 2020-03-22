from pytu.advanced.objectives import CrossEntropyLossND
import torch
import torch.nn as nn
import numpy as np

class MultiTaskLoss(nn.Module):

    def __init__(self, tissue_weights, class_weights, multitask_weight=[1.0, 1.0, 1.0], fix_multitask_weight=True, normalize_rate=100,
                 normalize_mode=True, learning_rate=0.5, epsilon=1e-5, use_cuda=True, use_geometrical_multitask=False,
                 use_uncertainity_multitask=False, reg_criterion=torch.nn.SmoothL1Loss(), **args):

        super().__init__()
        self.multitask_weight = multitask_weight
        self.fix_multitask_weight = fix_multitask_weight
        self.tissue_weights = tissue_weights
        self.class_weights = class_weights
        self.normalize_rate = normalize_rate
        self.normalize_mode = normalize_mode
        self.use_geometrical_multitask = use_geometrical_multitask
        self.use_uncertainity_multitask = use_uncertainity_multitask
        self.reg_criterion = reg_criterion
        self.learning_rate = learning_rate
        self.epsilon = epsilon

        if use_cuda:
            self.eta_seg = nn.Parameter(torch.ones(1).cuda())
            self.eta_class = nn.Parameter(torch.ones(1).cuda())
            self.eta_reg = nn.Parameter(torch.ones(1).cuda())
            self.register_buffer(name='running_seg_loss', tensor=torch.ones(1).cuda())
            self.register_buffer(name='running_class_loss', tensor=torch.ones(1).cuda())
            self.register_buffer(name='running_reg_loss', tensor=torch.ones(1).cuda())
            self.register_buffer(name='seg_loss_updated', tensor=torch.ones(1).cuda())
            self.register_buffer(name='class_loss_updated', tensor=torch.ones(1).cuda())
            self.register_buffer(name='reg_loss_updated', tensor=torch.ones(1).cuda())
        else:
            self.eta_seg = nn.Parameter(torch.ones(1))
            self.eta_class = nn.Parameter(torch.ones(1))
            self.eta_reg = nn.Parameter(torch.ones(1))
            self.register_buffer(name='running_seg_loss', tensor=torch.ones(1))
            self.register_buffer(name='running_class_loss', tensor=torch.ones(1))
            self.register_buffer(name='running_reg_loss', tensor=torch.ones(1))
            self.register_buffer(name='seg_loss_updated', tensor=torch.ones(1))
            self.register_buffer(name='class_loss_updated', tensor=torch.ones(1))
            self.register_buffer(name='reg_loss_updated', tensor=torch.ones(1))

    def forward(self, x, tgs, idx):

        x_s, x_c, x_r = x
        tgs_s, tgs_c, tgs_r = tgs

        # Segmentation Loss
        seg_criterion = CrossEntropyLossND(weight=self.tissue_weights)
        seg_loss = seg_criterion(x_s, tgs_s)

        # Classification Loss
        class_criterion = CrossEntropyLossND(weight=self.class_weights)
        class_loss = class_criterion(x_c, tgs_c)

        # Regression Loss
        reg_loss = self.reg_criterion(x_r, tgs_r)

        if self.use_geometrical_multitask:
            loss_combined = (torch.abs(seg_loss * class_loss * reg_loss))**(1/3)

        elif self.use_uncertainity_multitask:
            loss_combined  = 0.5 * (seg_loss * torch.exp(-self.eta_seg) + self.eta_seg)
            loss_combined += 0.5 * (class_loss * torch.exp(-self.eta_class) + self.eta_class)
            loss_combined += 0.5 * (reg_loss * torch.exp(-self.eta_reg) + self.eta_reg)

        else:

            if self.fix_multitask_weight:
                w1 = (self.multitask_weight[0] / (self.multitask_weight[0] + self.multitask_weight[1] +  self.multitask_weight[2]))
                w2 = (self.multitask_weight[1] / (self.multitask_weight[0] + self.multitask_weight[1] +  self.multitask_weight[2]))
                w3 = (1 - w1 - w2)

            else:
                w1 = np.random.uniform((self.multitask_weight[0] / (self.multitask_weight[0] + self.multitask_weight[1] +  self.multitask_weight[2])), 1)
                w2 = np.random.uniform((self.multitask_weight[1] / (self.multitask_weight[0] + self.multitask_weight[1] +  self.multitask_weight[2])), 1-float(w1))
                w3 = (1 - w1 - w2)

            if self.normalize_mode:
                self.running_seg_loss.data = (1 - self.learning_rate) * self.running_seg_loss.data + self.learning_rate * seg_loss.data
                self.running_class_loss.data = (1 - self.learning_rate) * self.running_class_loss.data + self.learning_rate * class_loss.data
                self.running_reg_loss.data = (1 - self.learning_rate) * self.running_reg_loss.data + self.learning_rate * reg_loss.data

                if idx > 0 and idx % self.normalize_rate == 0:
                    self.seg_loss_updated.data = self.running_seg_loss.data
                    self.class_loss_updated.data = self.running_class_loss.data
                    self.reg_loss_updated.data = self.running_reg_loss.data

                seg_loss_normalized = seg_loss / (self.seg_loss_updated + self.epsilon)
                class_loss_normalized = class_loss / (self.class_loss_updated + self.epsilon)
                reg_loss_normalized = reg_loss / (self.reg_loss_updated + self.epsilon)
                loss_combined = w1 * seg_loss_normalized + w2 * class_loss_normalized + w3 * reg_loss_normalized

            else:
                loss_combined = w1 * seg_loss + w2 * class_loss + w3 * reg_loss

        return loss_combined