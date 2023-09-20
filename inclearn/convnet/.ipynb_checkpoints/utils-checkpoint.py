import numpy as np
import torch
from torch import nn
from torch.optim import SGD, Adam
import torch.nn.functional as F
from inclearn.tools.metrics import ClassErrorMeter, AverageValueMeter
import json


def finetune_last_layer(
    logger,
    network,
    loader,
    n_class,
    nepoch=30,
    lr=0.1,
    scheduling=[15, 35],
    lr_decay=0.1,
    weight_decay=5e-4,
    loss_type="ce",
    temperature=5.0,
    test_loader=None,
    scheduler_type="cosine",
    optimizer_type="adam",
    mixup=False,
    alpha=0.5
):
    network.eval()
    #if hasattr(network.module, "convnets"):
    #    for net in network.module.convnets:
    #        net.eval()
    #else:
    #    network.module.convnet.eval()
    if optimizer_type == "adam":
        optim = Adam(network.module.classifier.parameters(), lr=lr, weight_decay=weight_decay)
    else:
        optim = SGD(network.module.classifier.parameters(), lr=lr, momentum=0.9, weight_decay=weight_decay)
        
    if "cos" in scheduler_type:
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optim, nepoch)
    else:
        scheduler = torch.optim.lr_scheduler.MultiStepLR(optim, scheduling, gamma=lr_decay)

    if loss_type == "ce":
        criterion = nn.CrossEntropyLoss()
    else:
        criterion = nn.BCEWithLogitsLoss()

    logger.info("Begin finetuning last layer")

    for i in range(nepoch):
        total_loss = 0.0
        total_correct = 0.0
        total_count = 0
        # print(f"dataset loader length {len(loader.dataset)}")
        for inputs, targets in loader:
            inputs, targets = inputs.cuda(), targets.cuda()
            if loss_type == "bce":
                targets = to_onehot(targets, n_class)
                
            if mixup:
                lam = np.random.beta(alpha, alpha)
                index = torch.randperm(inputs.size(0)).cuda()

                inputs = lam*inputs + (1-lam)*inputs[index,:]
                targets_a, targets_b = targets, targets[index]
                # targets_a = F.one_hot(targets_a, num_classes = n_class)
                # targets_b = F.one_hot(targets_b, num_classes = n_class)
                # targets = lam*targets_a + (1-lam)*targets_b
                
            outputs = network(inputs)['logit']
            _, preds = outputs.max(1)
            optim.zero_grad()
            if mixup:
                loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
            else:
                loss = criterion(outputs / temperature, targets)
            # loss = -(outputs.log_softmax(dim=-1) * targets).sum(dim=-1).mean()
            # loss = lam * criterion(outputs, targets_a) + (1-lam) * criterion(outputs, targets_b)
            loss.backward()
            optim.step()
            total_loss += loss * inputs.size(0)
            if mixup:
                total_correct += 0.0
            else:
                total_correct += (preds == targets).sum()
            total_count += inputs.size(0)

        if test_loader is not None:
            test_correct = 0.0
            test_count = 0.0
            with torch.no_grad():
                for inputs, targets in test_loader:
                    outputs = network(inputs.cuda())['logit']
                    _, preds = outputs.max(1)
                    test_correct += (preds.cpu() == targets).sum().item()
                    test_count += inputs.size(0)

        scheduler.step()
        if test_loader is not None:
            if mixup:
                logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct / total_count, test_correct / test_count))
            else:
                logger.info(
                "Epoch %d finetuning loss %.3f acc %.3f Eval %.3f" %
                (i, total_loss.item() / total_count, total_correct.item() / total_count, test_correct / test_count))

            acc = test_correct / test_count
        else:
            logger.info("Epoch %d finetuning loss %.3f acc %.3f" %
                        (i, total_loss.item() / total_count, total_correct.item() / total_count))
            
            acc = total_correct.item() / total_count
        
        loss = total_loss.item() / total_count

        if False:
            with open("./status.json", 'r') as f:
                status = json.load(f)
            status['epoch'] = i
            status['loss'] = loss
            status['acc'] = acc
            status['status'] = "Fine-tuning"
            status['completed'] = False
            with open("./status.json", 'w') as f:
                json.dump(status, f)

    return network


def extract_features(model, loader):
    targets, features = [], []
    model.eval()
    with torch.no_grad():
        for _inputs, _targets in loader:
            _inputs = _inputs.cuda()
            _targets = _targets.numpy()
            # _outputs = model(_inputs)
            _features = model.module(_inputs)['feature'].detach().cpu().numpy()
            features.append(_features)
            targets.append(_targets)

    return np.concatenate(features), np.concatenate(targets)


def calc_class_mean(network, loader, class_idx, metric):
    EPSILON = 1e-8
    features, targets = extract_features(network, loader)
    # norm_feats = features/(np.linalg.norm(features, axis=1)[:,np.newaxis]+EPSILON)
    # examplar_mean = norm_feats.mean(axis=0)
    examplar_mean = features.mean(axis=0)
    if metric == "cosine" or metric == "weight":
        examplar_mean /= (np.linalg.norm(examplar_mean) + EPSILON)
    return examplar_mean


def update_classes_mean(network, inc_dataset, n_classes, task_size, share_memory=None, metric="cosine", EPSILON=1e-8):
    loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                     inc_dataset.targets_inc,
                                     shuffle=False,
                                     share_memory=share_memory,
                                     mode="test")
    class_means = np.zeros((n_classes, network.module.features_dim))
    count = np.zeros(n_classes)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            feat = network(x.cuda())['feature']
            for lbl in torch.unique(y):
                class_means[lbl] += feat[y == lbl].sum(0).cpu().numpy()
                count[lbl] += feat[y == lbl].shape[0]
        for i in range(n_classes):
            class_means[i] /= count[i]
            if metric == "cosine" or metric == "weight":
                class_means[i] /= (np.linalg.norm(class_means) + EPSILON)
    return class_means

def cosine_similarity(a, b):
    return torch.mm(F.normalize(a, p=2, dim=-1), F.normalize(b, p=2, dim=-1).T)


def stable_cosine_distance(a, b, squared=True):
    """Computes the pairwise distance matrix with numerical stability."""
    mat = torch.cat([a, b])

    pairwise_distances_squared = torch.add(
        mat.pow(2).sum(dim=1, keepdim=True).expand(mat.size(0), -1),
        torch.t(mat).pow(2).sum(dim=0, keepdim=True).expand(mat.size(0), -1)
    ) - 2 * (torch.mm(mat, torch.t(mat)))

    # Deal with numerical inaccuracies. Set small negatives to zero.
    pairwise_distances_squared = torch.clamp(pairwise_distances_squared, min=0.0)

    # Get the mask where the zero distances are at.
    error_mask = torch.le(pairwise_distances_squared, 0.0)

    # Optionally take the sqrt.
    if squared:
        pairwise_distances = pairwise_distances_squared
    else:
        pairwise_distances = torch.sqrt(pairwise_distances_squared + error_mask.float() * 1e-16)

    # Undo conditionally adding 1e-16.
    pairwise_distances = torch.mul(pairwise_distances, (error_mask == False).float())

    # Explicitly set diagonals to zero.
    mask_offdiagonals = 1 - torch.eye(*pairwise_distances.size(), device=pairwise_distances.device)
    pairwise_distances = torch.mul(pairwise_distances, mask_offdiagonals)

    return pairwise_distances[:a.shape[0], a.shape[0]:]