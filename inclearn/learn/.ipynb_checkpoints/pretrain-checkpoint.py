import os.path as osp

import torch
import torch.nn.functional as F
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter
import numpy as np
import json

# import line_profiler
# import atexit
# profile = line_profiler.LineProfiler()
# atexit.register(profile.print_stats)


def _compute_loss(cfg, logits, targets, device, lam):

    if cfg["train_head"] == "sigmoid":
        n_classes = cfg["start_class"]
        onehot_targets = utils.to_onehot(targets, n_classes).to(device)
        loss = F.binary_cross_entropy_with_logits(logits, onehot_targets)
    elif cfg["train_head"] == "softmax":
        if cfg["mixup"] and lam != 0:
            loss = lam*F.cross_entropy(logits, targets[0]) + (1-lam)*F.cross_entropy(logits, targets[1])
        else:
            loss = F.cross_entropy(logits, targets)
    else:
        raise ValueError()
        
    if cfg["pretrain"]["gce"] == True:
        gce_topk = cfg["pretrain"]["gce_topk"]
        x = logits.cpu()
        y = targets.cpu()
        x1 = x.clone().cpu()
        x1[range(x1.size(0)), y] = -float("Inf")
        x_gt = x[range(x.size(0)), y].unsqueeze(1)
        x_topk = torch.topk(x1, gce_topk, dim=1)[0]  # 15 Negative classes to focus on, its a hyperparameter
        x_new = torch.cat([x_gt, x_topk], dim=1).to(device)
        loss += F.cross_entropy(x_new, torch.zeros(x_new.size(0)).long().to(device))
        
    return loss


def train(cfg, model, optimizer, device, train_loader):
    _loss = 0.0
    _adv_loss = 0.0
    _loc_loss = 0.0
    _ce_loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.train()
    for i, (inputs, targets) in enumerate(train_loader, start=1):
        # assert torch.isnan(inputs).sum().item() == 0
        optimizer.zero_grad()
        inputs, targets = inputs.to(device), targets.to(device)
        
        lam = 0
        if cfg["pretrain"]["mixup"]:
            alpha = cfg["pretrain"]["mixup_a"]
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs.size(0))
            
            inputs = lam*inputs + (1-lam)*inputs[index, :]
            targets_a, targets_b = targets, targets[index]
            targets = torch.stack((targets_a, targets_b))
            
        
        if cfg["jigsaw"] == True:
            inputs1, jigsaw_law = model.jigsaw_generator(inputs, cfg["jigsaw_patches"], cfg["RCM"])
            jigsaw = torch.cat((torch.zeros(inputs.shape[0], dtype=torch.int64), torch.ones(inputs.shape[0], dtype=torch.int64)), 0).clone().detach().to(device)
            inputs = torch.cat((inputs, torch.tensor(inputs1).to(device)), 0)
            targets = torch.cat((targets, targets), -1).to(device)
            
        outputs = model._parallel_network(inputs)
        logits = outputs["logit"]
        if accu is not None:
            if cfg["pretrain"]["mixup"]:
                accu.add(logits.detach(), targets[0])
            else:
                accu.add(logits.detach(), targets)

        loss = _compute_loss(cfg, logits, targets, device, lam)
        _ce_loss += loss
        
        if cfg["aux_open"] == True:
            open_targets = targets.clone()
            open_targets = [0 if i == 0 else 1 for i in open_targets].to(device)
            loss += F.cross_entropy(outputs['open_logit'], open_targets)

        if cfg["rotation"] == True:
            rotated_inputs = [inputs]
            angles = [torch.zeros(len(inputs))]
            
            for ang in range(1, 4):
                rotated_inputs.append(inputs.rot90(ang, [2, 3]))
                angles.append(torch.ones(len(inputs)) * ang)
            
            rotated_inputs = torch.cat(rotated_inputs)
            angles = torch.cat(angles).long().to(device)
            outputs = model._parallel_network(rotated_inputs, rotation=True)
            rot_loss = F.cross_entropy(outputs["rotations"], angles)
            loss += 0.1*rot_loss
            # print(rot_loss)
        
        if cfg["jigsaw"] == True:
            adv_loss = F.cross_entropy(outputs["jigsaw_adv"], jigsaw)
            loss += adv_loss
            _adv_loss += adv_loss
            
            patches = cfg["jigsaw_patches"] ** 2
            swap_law = []
            swap_law1 = [(i-(patches//2))/patches for i in range(patches)]
            swap_law2 = []
            for i in range(len(jigsaw_law)):
                swap_law.append(swap_law1)
                swap_law2.append([(j-(patches//2))/patches for j in jigsaw_law[i]])
            swap_law.extend(swap_law2)
            swap_law = torch.tensor(swap_law).to(device)
            loc_loss = F.l1_loss(outputs["mask"], swap_law)
            
            loss += loc_loss * int(cfg["pretrain"]["loc"])
            _loc_loss += loc_loss * int(cfg["pretrain"]["loc"])
        
        if torch.isnan(loss):
            import pdb

            pdb.set_trace()

        loss.backward()
        optimizer.step()
        _loss += loss
        # print("ce_loss: {}, adv_loss: {}, loc_loss: {}".format(round(_ce_loss.item() / i, 3), round(_adv_loss.item() / i , 3), round(_loc_loss.item() / i , 3)))

        

    return (
        round(_loss.item() / i, 3),
        round(accu.value()[0], 3),
    )


def test(cfg, model, device, test_loader):
    _loss = 0.0
    accu = ClassErrorMeter(accuracy=True)
    accu.reset()

    model.eval()
    with torch.no_grad():
        for i, (inputs, targets) in enumerate(test_loader, start=1):
            # assert torch.isnan(inputs).sum().item() == 0
            inputs, targets = inputs.to(device), targets.to(device)
            logits = model._parallel_network(inputs)['logit']
            if accu is not None:
                accu.add(logits.detach(), targets)
            loss = _compute_loss(cfg, logits, targets, device, 0)
            if torch.isnan(loss):
                import pdb
                pdb.set_trace()

            _loss = _loss + loss
    return round(_loss.item() / i, 3), round(accu.value()[0], 3)


def pretrain(cfg, ex, model, device, train_loader, test_loader, model_path):
    ex.logger.info(f"nb Train {len(train_loader.dataset)} Eval {len(test_loader.dataset)}")
    
    if cfg["pretrain"]["optimizer"] == "adam":
        optimizer = torch.optim.Adam(model._network.parameters(),
                                     lr=cfg["pretrain"]["lr"],
                                     weight_decay=cfg["pretrain"]["weight_decay"])
    else:
        optimizer = torch.optim.SGD(model._network.parameters(),
                                lr=cfg["pretrain"]["lr"],
                                momentum=0.9,
                                weight_decay=cfg["pretrain"]["weight_decay"])
    
    if "cos" in cfg["pretrain"]["scheduler"]:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, cfg["pretrain"]["epochs"])
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer,
                                                     cfg["pretrain"]["scheduling"],
                                                     gamma=cfg["pretrain"]["lr_decay"])
    
    test_loss, test_acc = float("nan"), float("nan")
    for e in range(cfg["pretrain"]["epochs"]):
        train_loss, train_acc = train(cfg, model, optimizer, device, train_loader)
        if e % 1 == 0:
            test_loss, test_acc = test(cfg, model, device, test_loader)
            ex.logger.info(
                "Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} | Eval loss: {}, Accu {}".format(
                    cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc, test_loss, test_acc))
        else:
            ex.logger.info("Pretrain Class {}, Epoch {}/{} => Clf Train loss: {}, Accu {} ".format(
                cfg["start_class"], e + 1, cfg["pretrain"]["epochs"], train_loss, train_acc))
        scheduler.step()

        if True:
            status = dict()
            status['task'] = 0
            status['epoch'] = e
            status['loss'] = train_loss
            status['acc'] = test_acc
            status['status'] = "Training"
            status['completed'] = False
            with open("./status.json", 'w') as f:
                json.dump(status, f)

    # if hasattr(model._network, "module"):
    #     torch.save(model._network.module.state_dict(), model_path)
    # else:
    #     torch.save(model._network.state_dict(), model_path)
