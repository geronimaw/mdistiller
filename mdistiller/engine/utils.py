import os
import torch
import torch.nn as nn
import numpy as np
import sys
import time
from tqdm import tqdm


class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count if self.count != 0 else 0


def validate(val_loader, distiller):
    batch_time, losses, top1, top5 = [AverageMeter() for _ in range(4)]
    criterion = nn.CrossEntropyLoss()
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))

    distiller.eval()
    with torch.no_grad():
        start_time = time.time()
        for idx, (image, target) in enumerate(val_loader):
            image = image.float()
            image = image.cuda(non_blocking=True)
            target = target.cuda(non_blocking=True)
            output = distiller(image=image)
            loss = criterion(output, target)
            acc1, acc5 = accuracy(output, target, topk=(1, 5))
            batch_size = image.size(0)
            losses.update(loss.cpu().detach().numpy().mean(), batch_size)
            top1.update(acc1[0], batch_size)
            top5.update(acc5[0], batch_size)

            # measure elapsed time
            batch_time.update(time.time() - start_time)
            start_time = time.time()
            msg = "Top-1:{top1.avg:.3f}| Top-5:{top5.avg:.3f}".format(
                top1=top1, top5=top5
            )
            pbar.set_description(log_msg(msg, "EVAL"))
            pbar.update()
    pbar.close()
    return top1.avg, top5.avg, losses.avg


def validate_hrnet(config, val_loader, model, criterion, criterion_kld, epoch = 0):
    
    def get_current_consistency_weight(current, rampup_length):
        # Consistency ramp-up from https://arxiv.org/abs/1610.02242
        if rampup_length == 0:
            return 1.0
        else:
            current = np.clip(current, 0.0, rampup_length)
            phase = 1.0 - current / rampup_length
            return float(np.exp(-5.0 * phase * phase))
        
    num_outputs = config.MODEL.N_STAGE if config.MODEL.MULTI else 1
    loss_by_stage = [AverageMeter() for i in range(num_outputs)]
    acc_by_stage = [AverageMeter() for i in range(num_outputs)]
    losses = AverageMeter()
    batch_time = AverageMeter()
    losses_hard = AverageMeter()
    losses_soft = AverageMeter()
    accs_by_stage = AverageMeter()
    teacher_losses = AverageMeter()
    losses_by_stage = AverageMeter()
    
    # switch to evaluate mode
    # model.eval()
    
    num_iter = len(val_loader)
    pbar = tqdm(range(num_iter))
    
    teacher_weight = config.TRAIN.TEACHER_WEIGHT
    kld_weight = config.TRAIN.KLD_WEIGHT
    cons_weight = get_current_consistency_weight(epoch, config.TRAIN.LENGTH)

    print("Validation\n---------------------")
    with torch.no_grad():
        end = time.time()
        for i, (input, target, target_weight, meta) in enumerate(val_loader):
            # compute outputs = teacher_out, stud_out
            outputs = model(input)
            target = target.cuda(non_blocking=True)
            target_weight = target_weight.cuda(non_blocking=True)
            batch_size = input.size(0)
            
            loss_hard = 0
            loss_soft = 0
            teacher_loss = 0

            if isinstance(outputs, list):

                if not config.MODEL.MULTI:
                    outputs = [ outputs[-1] ]   
                else:
                    kld_couples = config.LOSS.KLD_COUPLES
                    dist_to = [l[0] for l in kld_couples]

                for index, output in enumerate(outputs):                
                    if index == len(outputs) - 1:
                        teacher_loss = criterion(output, target, target_weight)
                        teacher_loss *= teacher_weight
                        stage_loss = teacher_loss
                    else:
                        ls = 0
                        lh = 0
                        
                        if index+1 in dist_to and config.LOSS.USE_MSE:
                            lh += criterion(output, target, target_weight)
                        
                        if config.LOSS.USE_KLD:
                            for index_dist_from in range(index+1, config.MODEL.N_STAGE):
                                if [index+1, index_dist_from+1] in kld_couples:
                                    ls += criterion_kld(output, outputs[index_dist_from], target_weight)
                    
                        ls *= kld_weight * cons_weight

                        loss_hard += lh
                        loss_soft += ls
                        stage_loss = lh + ls
                        
                    loss_by_stage[index].update(stage_loss, input.size(0))
                    _, avg_acc, cnt, pred = accuracy(output.detach().cpu().numpy(),
                                            target.detach().cpu().numpy())
                    acc_by_stage[index].update(avg_acc, cnt)
            else:
                raise ValueError("Model output is not a list")

        loss = loss_hard + teacher_loss + loss_soft
        losses.update(loss.cpu().detach().numpy().mean(), batch_size)
        losses_hard.update(loss_hard, batch_size)
        losses_soft.update(loss_soft, batch_size)
        teacher_losses.update(teacher_weight, batch_size)
        losses_by_stage.update(loss_by_stage, batch_size)
        accs_by_stage.update(acc_by_stage, batch_size)
        # measure elapsed time
        batch_time.update(time.time() - end)
        end = time.time()

        # measure elapsed time
        batch_time.update(time.time() - start_time)
        start_time = time.time()
        msg = "Test loss:{tloss.avg:.3f}| Test accuracy by stage:{tacc.avg:.3f}".format(
            tloss=loss.cpu().detach().numpy().mean(), tacc=acc_by_stage
        )
        pbar.set_description(log_msg(msg, "EVAL"))
        pbar.update()
    pbar.close()
    
    losses_dict = {
                "loss": losses,
                "loss_hard": losses_hard,
                "loss_soft": losses_soft,
                "teacher_loss": teacher_losses,
    }
    for i in range(num_outputs):
        losses_dict[f"loss_stage{i}"] = losses_by_stage[i]
        losses_dict[f"acc_stage{i}"] = accs_by_stage[i]
    return losses_dict


def log_msg(msg, mode="INFO"):
    color_map = {
        "INFO": 36,
        "TRAIN": 32,
        "EVAL": 31,
    }
    msg = "\033[{}m[{}] {}\033[0m".format(color_map[mode], mode, msg)
    return msg


def adjust_learning_rate(epoch, cfg, optimizer):
    steps = np.sum(epoch > np.asarray(cfg.SOLVER.LR_DECAY_STAGES))
    if steps > 0:
        new_lr = cfg.SOLVER.LR * (cfg.SOLVER.LR_DECAY_RATE**steps)
        for param_group in optimizer.param_groups:
            param_group["lr"] = new_lr
        return new_lr
    return cfg.SOLVER.LR


def accuracy(output, target, topk=(1,)):
    with torch.no_grad():
        maxk = max(topk)
        batch_size = target.size(0)
        _, pred = output.topk(maxk, 1, True, True)
        pred = pred.t()
        correct = pred.eq(target.reshape(1, -1).expand_as(pred))
        res = []
        for k in topk:
            correct_k = correct[:k].reshape(-1).float().sum(0, keepdim=True)
            res.append(correct_k.mul_(100.0 / batch_size))
        return res


def save_checkpoint(obj, path):
    with open(path, "wb") as f:
        torch.save(obj, f)


def load_checkpoint(path):
    with open(path, "rb") as f:
        return torch.load(f, map_location="cpu")
    
