import torch
import numpy as np
import torch.nn as nn
import torch.nn.functional as F

from ._base import Distiller
from mdistiller.engine.evaluate import accuracy
from mdistiller.engine.losses import JointsMSELoss, JointsKLDLoss
from mdistiller.engine.utils import validate_hrnet


def kd_loss(logits_student, logits_teacher, temperature):
    log_pred_student = F.log_softmax(logits_student / temperature, dim=1)
    pred_teacher = F.softmax(logits_teacher / temperature, dim=1)
    loss_kd = F.kl_div(log_pred_student, pred_teacher, reduction="none").sum(1).mean()
    loss_kd *= temperature**2
    return loss_kd


class HROKD(Distiller):
    """Distilling the Knowledge in a Neural Network"""

    def __init__(self, student, teacher, cfg):
        super(HROKD, self).__init__(student, teacher)
        self.cfg = cfg
        self.temperature = cfg.KD.TEMPERATURE
        self.ce_loss_weight = cfg.KD.LOSS.CE_WEIGHT
        self.kd_loss_weight = cfg.KD.LOSS.KD_WEIGHT
        self.criterion_kld = JointsKLDLoss().cuda()
        self.criterion = JointsMSELoss(use_target_weight=cfg.LOSS.USE_TARGET_WEIGHT).cuda()
        
        self.num_outputs = self.cfg.MODEL.N_STAGE if self.cfg.MODEL.MULTI else 1
        self.roles = ['Teacher_' if i == self.num_outputs-1 else f'Student{i+1}'
                      for i in range(self.num_outputs)]

    def get_current_consistency_weight(self, current, rampup_length):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
            
    def forward_train(self, input, target, target_weight, epoch):#, **kwargs):
        outputs = self.student(input)

        target = target.cuda(non_blocking=True)
        target_weight = target_weight.cuda(non_blocking=True)
        cons_weight = self.get_current_consistency_weight(epoch, self.cfg.TRAIN.LENGTH)

        hard_loss = 0
        soft_loss = 0
        teacher_loss = 0
        loss_by_stage = [0 for i in range(self.num_outputs)]
        acc_by_stage = [0 for i in range(self.num_outputs)]

        if isinstance(outputs, list):
            if not self.cfg.MODEL.MULTI:
                outputs = [ outputs[-1] ]   
            else:
                kld_couples = self.cfg.LOSS.KLD_COUPLES
                dist_to = [ couple[0] for couple in kld_couples]

            for index, output in enumerate(outputs):                
                if index == len(outputs) - 1:
                    teacher_loss = self.criterion(output, target, target_weight)
                    teacher_loss *= self.cfg.TRAIN.TEACHER_WEIGHT
                    stage_loss = teacher_loss
                
                else:
                    ls = 0
                    lh = 0
                    
                    if index+1 in dist_to and self.cfg.LOSS.USE_MSE:
                        lh += self.criterion(output, target, target_weight)
                    
                    if self.cfg.LOSS.USE_KLD:
                        for index_dist_from in range(index+1, self.cfg.MODEL.N_STAGE):
                            if [index+1, index_dist_from+1] in kld_couples:
                                ls += self.criterion_kld(output, outputs[index_dist_from], target_weight)
                
                    ls *= self.cfg.TRAIN.KLD_WEIGHT * cons_weight

                    hard_loss += lh
                    soft_loss += ls
                    stage_loss = lh + ls
                    
                loss_by_stage[index] = stage_loss
                _, stage_accuracy, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                                        target.detach().cpu().numpy())
                acc_by_stage[index] = stage_accuracy
        else:
            raise ValueError("Model output is not a list")

        loss = hard_loss + teacher_loss + soft_loss

        losses_dict = {
            "loss": loss,
            "hard_loss": hard_loss,
            "soft_loss": soft_loss,
            "teacher_loss": teacher_loss,
            }
        for i in range(self.num_outputs):
            losses_dict[f"loss_stage{i+1}"] = loss_by_stage[i]
            losses_dict[f"acc_stage{i+1}"] = acc_by_stage[i]
    
        return losses_dict

    def forward_test(self, input, target, target_weight, epoch):
        num_outputs = self.cfg.MODEL.N_STAGE if self.cfg.MODEL.MULTI else 1
        loss_by_stage = [0 for i in range(num_outputs)]
        acc_by_stage = [0 for i in range(num_outputs)]
        
        teacher_weight = self.cfg.TRAIN.TEACHER_WEIGHT
        kld_weight = self.cfg.TRAIN.KLD_WEIGHT
        cons_weight = self.get_current_consistency_weight(epoch, self.cfg.TRAIN.LENGTH)

        with torch.no_grad():
            # compute outputs = teacher_out, stud_out
            outputs = self.student(input)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            batch_size = input.size(0)
            
            hard_loss = 0
            soft_loss = 0
            teacher_loss = 0

            if isinstance(outputs, list):

                if not self.cfg.MODEL.MULTI:
                    outputs = [ outputs[-1] ]   
                else:
                    kld_couples = self.cfg.LOSS.KLD_COUPLES
                    dist_to = [l[0] for l in kld_couples]

                for index, output in enumerate(outputs):                
                    if index == len(outputs) - 1:
                        teacher_loss = self.criterion(output, target, target_weight)
                        teacher_loss *= teacher_weight
                        stage_loss = teacher_loss
                    else:
                        ls = 0
                        lh = 0
                        
                        if index+1 in dist_to and self.cfg.LOSS.USE_MSE:
                            lh += self.criterion(output, target, target_weight)
                        
                        if self.cfg.LOSS.USE_KLD:
                            for index_dist_from in range(index+1, self.cfg.MODEL.N_STAGE):
                                if [index+1, index_dist_from+1] in kld_couples:
                                    ls += self.criterion_kld(output, outputs[index_dist_from], target_weight)
                    
                        ls *= kld_weight * cons_weight

                        hard_loss += lh
                        soft_loss += ls
                        stage_loss = lh + ls
                        
                    loss_by_stage[index] = stage_loss
                    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
                    acc_by_stage[index] = avg_acc
            else:
                raise ValueError("Model output is not a list")

            loss = hard_loss + teacher_loss + soft_loss
            
            losses_dict = {
                "loss": loss,
                "hard_loss": hard_loss,
                "soft_loss": soft_loss,
                "teacher_loss": teacher_loss,
            }
        for i in range(self.num_outputs):
            losses_dict[f"loss_stage{i+1}"] = loss_by_stage[i]
            losses_dict[f"acc_stage{i+1}"] = acc_by_stage[i]
        return losses_dict

    def forward(self, **kwargs):
        if self.training:
            return self.forward_train(**kwargs)
        return self.forward_test(**kwargs)
