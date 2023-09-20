import random
from copy import deepcopy
import numpy as np
import datetime
import os

import torch

from inclearn.tools.metrics import ClassErrorMeter

from sklearn.metrics import precision_recall_fscore_support, precision_recall_curve, roc_auc_score
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt
from statistics import mean, median


def get_date():
    return datetime.datetime.now().strftime("%Y%m%d")


def to_onehot(targets, n_classes):
    if not hasattr(targets, "device"):
        targets = torch.from_numpy(targets)
    onehot = torch.zeros(targets.shape[0], n_classes).to(targets.device)
    onehot.scatter_(dim=1, index=targets.long().view(-1, 1), value=1.0)
    return onehot


def get_class_loss(network, cur_n_cls, loader):
    class_loss = torch.zeros(cur_n_cls)
    n_cls_data = torch.zeros(cur_n_cls)  # the num of imgs for cls i.
    EPS = 1e-10
    task_size = 10
    network.eval()
    for x, y in loader:
        x, y = x.cuda(), y.cuda()
        preds = network(x)['logit'].softmax(dim=1)
        # preds[:,-task_size:] = preds[:,-task_size:].softmax(dim=1)
        for i, lbl in enumerate(y):
            class_loss[lbl] = class_loss[lbl] - (preds[i, lbl] + EPS).detach().log().cpu()
            n_cls_data[lbl] += 1
    class_loss = class_loss / n_cls_data
    return class_loss


def get_featnorm_grouped_by_class(network, cur_n_cls, loader):
    """
    Ret: feat_norms: list of list
            feat_norms[idx] is the list of feature norm of the images for class idx.
    """
    feats = [[] for i in range(cur_n_cls)]
    feat_norms = np.zeros(cur_n_cls)
    network.eval()
    with torch.no_grad():
        for x, y in loader:
            x = x.cuda()
            feat = network(x)['feature'].cpu()
            for i, lbl in enumerate(y):
                feats[lbl].append(feat[y == lbl])
    for i in range(len(feats)):
        if len(feats[i]) != 0:
            feat_cls = torch.cat((feats[i]))
            feat_norms[i] = torch.norm(feat_cls, p=2, dim=1).mean().data.numpy()
    return feat_norms


def set_seed(seed):
    print("Set seed", seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True  # This will slow down training.
    torch.backends.cudnn.benchmark = False


def display_weight_norm(logger, network, increments, tag):
    weight_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(network.module.classifier.weight.shape[0]):
        norm = torch.norm(network.module.classifier.weight[idx].data, p=2).item()
        for i in range(len(weight_norms)):
            if idx < increments[i]:
                break
        weight_norms[i].append(round(norm, 3))
    avg_weight_norm = []
    for idx in range(len(weight_norms)):
        avg_weight_norm.append(round(np.array(weight_norms[idx]).mean(), 3))
    logger.info("%s: Weight norm per class %s" % (tag, str(avg_weight_norm)))


def display_feature_norm(logger, network, loader, n_classes, increments, tag, return_norm=False):
    avg_feat_norm_per_cls = get_featnorm_grouped_by_class(network, n_classes, loader)
    feature_norms = [[] for _ in range(len(increments))]
    increments = np.cumsum(np.array(increments))
    for idx in range(len(avg_feat_norm_per_cls)):
        for i in range(len(feature_norms)):
            if idx < increments[i]:  #Find the mapping from class idx to step i.
                break
        feature_norms[i].append(round(avg_feat_norm_per_cls[idx], 3))
    avg_feature_norm = []
    for idx in range(len(feature_norms)):
        avg_feature_norm.append(round(np.array(feature_norms[idx]).mean(), 3))
    logger.info("%s: Feature norm per class %s" % (tag, str(avg_feature_norm)))
    if return_norm:
        return avg_feature_norm
    else:
        return


def check_loss(loss):
    return not bool(torch.isnan(loss).item()) and bool((loss >= 0.0).item())


def compute_accuracy(ypred, ytrue, increments, n_classes):
    all_acc = {"top1": {}, "top5": {}}
    topk = 5 if n_classes >= 5 else n_classes
    ncls = np.unique(ytrue).shape[0]
    if topk > ncls:
        topk = ncls
    all_acc_meter = ClassErrorMeter(topk=[1, topk], accuracy=True)
    all_acc_meter.add(ypred, ytrue)
    all_acc["top1"]["total"] = round(all_acc_meter.value()[0], 3)
    all_acc["top5"]["total"] = round(all_acc_meter.value()[1], 3)
    # all_acc["total"] = round((ypred.argmax(1) == ytrue).sum() / len(ytrue), 3)

    # for class_id in range(0, np.max(ytrue), task_size):
    start, end = 0, 0
    for i in range(len(increments)):
        if increments[i] <= 0:
            pass
        else:
            start = end
            end += increments[i]

            idxes = np.where(np.logical_and(ytrue >= start, ytrue < end))[0]
            topk_ = 5 if increments[i] >= 5 else increments[i]
            ncls = np.unique(ytrue[idxes]).shape[0]
            if topk_ > ncls:
                topk_ = ncls
            cur_acc_meter = ClassErrorMeter(topk=[1, topk_], accuracy=True)
            cur_acc_meter.add(ypred[idxes], ytrue[idxes])
            top1_acc = (ypred[idxes].argmax(1) == ytrue[idxes]).sum() / idxes.shape[0] * 100
            if start < end:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end - 1).rjust(2, "0"))
            else:
                label = "{}-{}".format(str(start).rjust(2, "0"), str(end).rjust(2, "0"))
            all_acc["top1"][label] = round(top1_acc, 3)
            all_acc["top5"][label] = round(cur_acc_meter.value()[1], 3)
            # all_acc[label] = round((ypred[idxes] == ytrue[idxes]).sum() / len(idxes), 3)

    return all_acc

def compute_top1_accuracy(ypred, ytrue, start, end, n_classes):
    
    idxes = np.where(np.logical_and(ytrue >= start, ytrue < end))[0]
    top1_acc = (ypred[idxes].argmax(1) == ytrue[idxes]).sum() / idxes.shape[0] * 100
    
    return top1_acc

def compute_top1_accuracy_woopen(ypred, ytrue, start, end, n_classes):
    
    idxes = np.where(np.logical_and(ytrue >= start, ytrue < end))[0]
    top1_acc = (ypred[idxes].argmax(1) == ytrue[idxes]).sum()
    others_idxes = np.where(ytrue == 0)[0]
    others_acc = (ypred[others_idxes].argmax(1) == ytrue[others_idxes]).sum()
    acc = (top1_acc - others_acc) / (idxes.shape[0]-others_acc) * 100
    
    # print(others_acc)
    
    return acc

def compute_tp_fp(ypred, ytrue, start, end, n_classes):
    
    idxes = np.where(np.logical_and(ytrue > start, ytrue < end))[0]
    tp = (ypred[idxes].argmax(1) == ytrue[idxes]).sum()
    fp = (ypred[idxes].argmax(1) != ytrue[idxes]).sum()
    fp_to_others = (ypred[idxes].argmax(1) == 0).sum()
    others_idxes = np.where(ytrue == 0)[0]
    tp_others = (ypred[others_idxes].argmax(1) == ytrue[others_idxes]).sum()
    fp_others = (ypred[others_idxes].argmax(1) != ytrue[others_idxes]).sum()
    
    return tp, fp, fp_to_others, tp_others, fp_others

def per_class_performance(ypred, ytrue):
    
    per_class_stats = {"Precision": {}, "Recall": {}, "Fscore": {}, }
    p, r, f, s = precision_recall_fscore_support(ytrue, ypred.argmax(1))
    
    print(s)
    
    for i in range(len(p)):
        per_class_stats["Precision"][str(i)] = round(p[i]*100, 3)
        per_class_stats["Recall"][str(i)] = round(r[i]*100, 3)
        per_class_stats["Fscore"][str(i)] = round(f[i]*100, 3)
    
    per_class_stats["Precision"]["avg"] = round(sum(p)*100/len(p), 3)
    per_class_stats["Recall"]["avg"] = round(sum(r)*100/len(r), 3)
    per_class_stats["Fscore"]["avg"] = round(sum(f)*100/len(f), 3)
    
    return per_class_stats

def make_confusion_matrix(ypred, ytrue, task_i, use_name=False):
    
    # classes_names = ['Unknown','I-Nothing', 'I-Others', 'E-M1-Al Residue', 'I-Dust', 'I-ITo1-Hole', 'T-Fiber', 'I-Sand', 'I-Glass',
    #                'E-AS','I-Oil', 'E-ITO1-Hole', 'P-ITO1-Residue', 'I-Laser', 'P-AS-No', 'I-ITO1-Residue', 'I-M2-Crack',
    #                'E-M2-Residue','T-Brush', 'T-AS-Residue', 'T-AS-Partical small', 'E-M2-PR-Residue', 'T-M2-Particle',
    #                'P-M2-Residue', 'P-AS','P-M1-Residue', 'T-AS-SiN Hole', 'P-M2-Open']
    classes_names = ['Unknown', 
                     'I-Nothing', 
                     'I-Others',
                    'E-M1-Al Residue',
                    'I-Dust',
                    'T-ITO1-Hole',
                    'T-Fiber',
                    'I-Sand',
                    'I-Glass',
                    'E-AS',
                    'I-Oil',
                    'E-ITO1-Hole',
                    'P-ITO1-Residue',
                    'I-Laser',
                    'P-AS-No',
                    'T-ITO1-Residue',
                    'E-M2-Residue',
                    'P-M2-Residue',
                    'T-Brush',
                    'T-AS-SiN Hole',
                    'T-AS-Partical small',
                    'P-M1-Residue',
                    'T-AS-Residue',
                    'E-M2-PR-Residue',
                    'T-M2-Particle',
                    'P-AS',
                    'I-M2-Crack',
                    'P-M2-Open']
    
    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    _, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    
    cm = confusion_matrix(ytrue, pred)
    
    cf_name = os.path.join("./logs", str(task_i))
    
    print('Confusion matrix, without normalization')

    print(cm.shape)
    f = plt.figure(figsize=(20, 15))
    plt.imshow(cm, interpolation='nearest', cmap=plt.cm.Blues)
    plt.title('Confusion matrix')
    plt.colorbar()
    tick_marks = np.arange(len(cm))
    if use_name == True:
        classes = classes_names[:len(cm)]
    else:
        classes = np.arange(1, len(cm)+1)
    plt.xticks(tick_marks, classes, rotation=60, fontsize=14)
    plt.yticks(tick_marks, classes, fontsize=14)

    fmt = 'd'
    thresh = cm.max() / 2.
    for i in (range(cm.shape[0])):
        for j in (range(cm.shape[1])):
            plt.text(j, i, format(cm[i, j], fmt), fontsize=14, horizontalalignment="center", color="white" if cm[i, j] > thresh else "black")

    plt.tight_layout()
    plt.ylabel('True label')
    plt.xlabel('Predicted label')
    plt.savefig(cf_name)

# First, use test set to get the threshold of each classes
# Second, re-inference the test set
def compute_openset_acc(vpred, vtrue, ypred, ytrue, alpha, PlusOne=False, method="none", Rmethods="min", Wmethods="max"):
    
    diff = 0 if PlusOne else 1
    
    openacc = {'openacc': {}}
    
    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    if "softmax" in method:
        output = torch.nn.functional.softmax(output, dim=1)
    elif "l2" in method:
        output = torch.nn.functional.normalize(output)
    elif "min-max" in method:
        v_max = output.amax(dim=1, keepdim=True)
        v_min = output.amin(dim=1, keepdim=True)
        output = (output - v_min) / (v_max - v_min)
    elif "z-score" in method:
        v_means = output.mean(dim=1, keepdim=True)
        v_stds = output.std(dim=1, keepdim=True)
        output = (output - v_means) / v_stds
    score, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    
    v_output = torch.tensor(vpred)
    if "softmax" in method:
        v_output = torch.nn.functional.softmax(v_output, dim=1)
    elif "l2" in method:
        v_output = torch.nn.functional.normalize(v_output)
    elif "min-max" in method:
        v_max = v_output.amax(dim=1, keepdim=True)
        v_min = v_output.amin(dim=1, keepdim=True)
        v_output = (v_output - v_min) / (v_max - v_min)
    elif "z-score" in method:
        v_means = v_output.mean(dim=1, keepdim=True)
        v_stds = v_output.std(dim=1, keepdim=True)
        v_output = (v_output - v_means) / v_stds
    v_score, v_pred = v_output.topk(topk, 1, True, True)
    v_pred = v_pred.t()
    v_pred = v_pred.numpy().reshape(-1)
    
    R_thresholds = np.zeros(nb_classes)
    W_thresholds = np.zeros(nb_classes)
    thresholds = np.zeros(nb_classes)
    
    R_list = [list() for i in range(nb_classes+1)]
    W_list = [list() for i in range(nb_classes+1)]
    
    for i in range(len(v_pred)):
        if v_pred[i] == vtrue[i]:
            R_list[v_pred[i]].append(v_score[i].item())
        else:
            W_list[v_pred[i]].append(v_score[i].item())
    
    if "median" in Rmethods:
        for i in range(len(R_thresholds)):
            if len(R_list[i]) != 0:
                R_thresholds[i] = median(R_list[i])
            else:
                R_thresholds[i] = 0
    elif "mean" in Rmethods:
        for i in range(len(R_thresholds)):
            if len(R_list[i]) != 0:
                R_thresholds[i] = mean(R_list[i])
            else:
                R_thresholds[i] = 0
    elif "min" in Rmethods:
        for i in range(len(R_thresholds)):
            if len(R_list[i]) != 0:
                R_thresholds[i] = min(R_list[i])
            else:
                R_thresholds[i] = 0
    elif "max" in Rmethods:
        for i in range(len(R_thresholds)):
            if len(R_list[i]) != 0:
                R_thresholds[i] = max(R_list[i])
            else:
                R_thresholds[i] = 0
    
    if "median" in Wmethods:
        for i in range(len(W_thresholds)):
            if len(W_list[i]) != 0:
                W_thresholds[i] = median(W_list[i])
            else:
                W_thresholds[i] = 0
    elif "mean" in Wmethods:
        for i in range(len(W_thresholds)):
            if len(W_list[i]) != 0:
                W_thresholds[i] = mean(W_list[i])
            else:
                W_thresholds[i] = 0
    elif "min" in Wmethods:
        for i in range(len(W_thresholds)):
            if len(W_list[i]) != 0:
                W_thresholds[i] = min(W_list[i])
            else:
                W_thresholds[i] = 0
    elif "max" in Wmethods:
        for i in range(len(W_thresholds)):
            if len(W_list[i]) != 0:
                W_thresholds[i] = max(W_list[i])
            else:
                W_thresholds[i] = 0
            
    thresholds = alpha * R_thresholds + (1-alpha) * W_thresholds

    # re-inference
    for i in range(len(score)):
        if score[i] < thresholds[pred[i]]:
            pred[i] = 0
        else:
            pred[i] += diff
    
    # Acc
    top1_acc = (pred == ytrue).sum() / len(ytrue) * 100
    openacc['openacc'] = round(top1_acc, 3)
    
    o_pred = np.eye(nb_classes+1)[pred]
    
    return thresholds, openacc, o_pred

def threshold_status(ypred, ytrue, PlusOne=False, method="none"):
    
    diff = 0 if PlusOne else 1
    
    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    if "softmax" in method:
        output = torch.nn.functional.softmax(output, dim=1)
    elif "l2" in method:
        output = torch.nn.functional.normalize(output)
    elif "min-max" in method:
        v_max = output.amax(dim=1, keepdim=True)
        v_min = output.amin(dim=1, keepdim=True)
        output = (output - v_min) / (v_max - v_min)
    elif "z-score" in method:
        v_means = output.mean(dim=1, keepdim=True)
        v_stds = output.std(dim=1, keepdim=True)
        output = (output - v_means) / v_stds
        
    score, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    
    
    from statistics import mean, median
    
    R_list = [list() for i in range(nb_classes+1)]
    W_list = [list() for i in range(nb_classes+1)]
    R_status_list = [list() for i in range(4)]
    W_status_list = [list() for i in range(4)]
    
    for i in range(len(pred)):
        if pred[i] == ytrue[i]:
            R_list[pred[i]].append(score[i].item())
        else:
            W_list[pred[i]].append(score[i].item())
        
    #max min mean median
    for i in range(len(R_list)):
        if len(R_list[i]) != 0:
            R_status_list[0].append(max(R_list[i]))
            R_status_list[1].append(min(R_list[i]))
            R_status_list[2].append(median(R_list[i]))
            R_status_list[3].append(mean(R_list[i]))
        else:
            R_status_list[0].append(0)
            R_status_list[1].append(0)
            R_status_list[2].append(0)
            R_status_list[3].append(0)
    
    for i in range(len(W_list)):
        if len(W_list[i]) != 0:
            W_status_list[0].append(max(W_list[i]))
            W_status_list[1].append(min(W_list[i]))
            W_status_list[2].append(median(W_list[i]))
            W_status_list[3].append(mean(W_list[i]))
        else:
            W_status_list[0].append(0)
            W_status_list[1].append(0)
            W_status_list[2].append(0)
            W_status_list[3].append(0)
    
    return [R_status_list, W_status_list]
    
def roc_thresholds(vpred, vtrue, ypred, ytrue, PlusOne=False):
    
    diff = 0 if PlusOne else 1
    
    from sklearn.metrics import roc_curve
    nb_classes = len(np.unique(ytrue))
    thresholds = np.zeros(nb_classes)
    # print(vpred[:, 0])
    vpred = torch.tensor(vpred)
    vpred = torch.nn.functional.normalize(vpred)
    vpred = vpred.numpy()
    for i in range(nb_classes):
        label = list()
        score = list()
        for j in range(len(vpred)):
            if vpred[j].argmax(0) == i:
                score.append(vpred[j][i])
                if i == vtrue[j]:
                    label.append(1)
                else:
                    label.append(0)
        # print(label)
        # print(score)
        # print(vpred[0].argmax(0))
        # fpr, tpr, thres = roc_curve(label, vpred[:, i])
        if len(label) != 0:
            precision, recall, thr = precision_recall_curve(label, score)
        # print(fpr, tpr, thres)
        # thresholds[i] = sorted(list(zip(np.abs(tpr - fpr), thres)), key=lambda s: s[0], reverse=True)[0][1]
            fscore = recall * precision * 2 / (recall + precision)
            thresholds[i] = thr[fscore.argmax(0)]
        else:
            thresholds[i] = 0
    
    openacc = {'openacc': {}}
    
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    score, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    
    # re-inference
    for i in range(len(score)):
        if score[i] < thresholds[pred[i]]:
            pred[i] = 0
        else:
            pred[i] += diff
    
    # Acc
    top1_acc = (pred == ytrue).sum() / len(ytrue) * 100
    openacc['openacc'] = round(top1_acc, 3)
    
    o_pred = np.eye(nb_classes+1)[pred]
    
    return thresholds, openacc, o_pred

def fscore_thresholds(vpred, vtrue, ypred, ytrue, PlusOne=False, methods="avg", biased=0.5):
    diff = 0 if PlusOne else 1
    mechanism = "l2"
    
    nb_classes = len(np.unique(ytrue))
    vpred = torch.tensor(vpred)
    if mechanism == "l2":
        vpred = torch.nn.functional.normalize(vpred)
    elif mechanism == "softmax":
        vpred = torch.nn.functional.softmax(vpred, dim=1)
    elif "zscore":
        v_means = output.mean(dim=1, keepdim=True)
        v_stds = output.std(dim=1, keepdim=True)
        vpred = (vpred - vmeans) / v_stds

    vpred = vpred.numpy()
    
    score_list = [list() for i in range(nb_classes)]
    label_list = [list() for i in range(nb_classes)]
    for i in range(len(vpred)):
        max_score = vpred[i].argmax(0)
        score_list[max_score].append(vpred[i][max_score])
        if vtrue[i] == max_score:
            label_list[max_score].append(1)
        else:
            label_list[max_score].append(0)
    
    thresholds = np.zeros(nb_classes)
    for i in range(nb_classes):
        
        if len(label_list[i]) <= 3:
            precision = [0, 0]
            recall = [0, 0]
            thres = [0]
        else:
            precision, recall, thres = precision_recall_curve(label_list[i], score_list[i])
        
        p = np.array(precision)
        r = np.array(recall)
        if "fscore" in methods:
            f = np.nan_to_num(r*p*2/(r+p))
        elif "avg" in methods:
            f = p * biased + r * (1 - biased)
        index = np.argmax(f)
        # print(index)
        
        if index == 0:
            thresholds[i] = 0
        else:
            thresholds[i] = thres[index]
    
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    if mechanism == "l2":
        output = torch.nn.functional.normalize(output)
    elif mechanism == "softmax":
        output = torch.nn.functional.softmax(output, dim=1)
    elif "zscore":
        y_means = output.mean(dim=1, keepdim=True)
        y_stds = output.std(dim=1, keepdim=True)
        output = (output - vmeans) / v_stds
    score, pred = output.topk(topk, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    for i in range(len(score)):
        if score[i] < thresholds[pred[i]]:
            pred[i] = 0
            # print(123)
        else:
            pred[i] += diff
        
    openacc = {'openacc': {}}
    top1_acc = (pred == ytrue).sum() / len(ytrue) * 100
    openacc['openacc'] = round(top1_acc, 3)
    o_pred = np.eye(nb_classes)[pred]
    
    return openacc, o_pred

def compute_threshold(vpred, vtrue, PlusOne=False, methods="avg", biased=0.5):
    diff = 0 if PlusOne else 1
    
    nb_classes = len(np.unique(vtrue))
    vpred = torch.tensor(vpred)
    vpred = torch.nn.functional.normalize(vpred)
    vpred = vpred.numpy()
    
    score_list = [list() for i in range(nb_classes)]
    label_list = [list() for i in range(nb_classes)]
    for i in range(len(vpred)):
        max_score = vpred[i].argmax(0)
        score_list[max_score].append(vpred[i][max_score])
        if vtrue[i] == max_score:
            label_list[max_score].append(1)
        else:
            label_list[max_score].append(0)
    
    thresholds = np.zeros(nb_classes)
    for i in range(nb_classes):
        
        if len(label_list[i]) <= 3:
            precision = [0, 0]
            recall = [0, 0]
            thres = [0]
        else:
            precision, recall, thres = precision_recall_curve(label_list[i], score_list[i])
        
        p = np.array(precision)
        r = np.array(recall)
        if "fscore" in methods:
            f = np.nan_to_num(r*p*2/(r+p))
        elif "avg" in methods:
            f = p * biased + r * (1 - biased)
        index = np.argmax(f)
        # print(index)
        
        if index == 0:
            thresholds[i] = 0
        else:
            thresholds[i] = thres[index]
            
    return thresholds
        
def compute_by_thresholds(ypred, ytrue, thresholds):

    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    output = torch.nn.functional.normalize(output)
    score, pred = output.topk(topk, 1, True, True)
    score_2, pred_2 = output.topk(2, 1, True, True)
    pred = pred.t()
    pred = pred.numpy().reshape(-1)
    pred_2 = pred_2.numpy()
    for i in range(len(score)):
        # if pred[i] == 0:
        #     if score_2[i][1] >= float(thresholds[pred_2[i][1]]):
        #         pred[i] = pred_2[i][1]
        # else:
        #     if score[i] < float(thresholds[pred[i]]):
        #         pred[i] = 0
        if score[i] < float(thresholds[pred[i]]):
            pred[i] = 0
    
    openacc = {'openacc': {}}
    top1_acc = (pred == ytrue).sum() / len(ytrue) * 100
    openacc['openacc'] = round(top1_acc, 3)
    o_pred = np.eye(nb_classes)[pred]

    return openacc, o_pred

def compute_uosr_auroc(ypred, ytrue):

    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    # output = torch.nn.functional.normalize(output)
    output = torch.nn.functional.softmax(output, dim=1)
    score, pred = output.topk(topk, 1, True, True)
    # pred = pred.t()
    # pred = pred.numpy().reshape(-1)
    score = output[:, 0].numpy().reshape(-1)
    c = 0
    for i in range(len(score)):
        if pred[i] == ytrue[i] and ytrue[i] != 0:
            pred[i] = 0
        else:
            pred[i] = 1
            c += 1
    print(c)
    
    auc = roc_auc_score(pred, score)
    print(auc)

    return auc * 100
            
def compute_osr_auroc(ypred, ytrue):

    nb_classes = len(np.unique(ytrue))
    topk = min(1, nb_classes)
    output = torch.tensor(ypred)
    # output = torch.nn.functional.normalize(output)
    output = torch.nn.functional.softmax(output, dim=1)
    # score, pred = output.topk(topk, 1, True, True)
    # pred = pred.t()
    # pred = pred.numpy().reshape(-1)
    score = output[:, 0].numpy().reshape(-1)
    pred = list()
    c = 0
    for i in range(len(score)):
        if ytrue[i] != 0:
            pred.append(0)
        else:
            pred.append(1)
            c += 1
    print(c)
    
    auc = roc_auc_score(pred, score)
    print(auc)

    return auc * 100

# Due to the dataloader did't use shuffle, we can get the image by the index of the csv
def get_wrong_images_index(ypred, ytrue):
    
    index_list = list()
    score_list = list()
    
    t_pred = torch.tensor(ypred)
    t_pred = torch.nn.functional.normalize(t_pred)
    t_pred = t_pred.numpy()
    
    for i in range(len(ytrue)):
        if ypred[i].argmax(0) != ytrue[i]:
            
            index_list.append([i, ytrue[i], ypred[i].argmax(0)])
            tmp_list = list()
            tmp_list.append(i)
            tmp_list.append(ytrue[i])
            tmp_list.extend(t_pred[i])
            score_list.append(tmp_list)
    
    return index_list, score_list

def get_score_list(ypred, ytrue):
    
    index_list = list()
    score_list = list()
    
    t_pred = torch.tensor(ypred)
    t_pred = torch.nn.functional.normalize(t_pred)
    t_pred = t_pred.numpy()
    
    for i in range(len(ytrue)):
        if ypred[i].argmax(0) != ytrue[i]:
            
            index_list.append([i, ytrue[i], ypred[i].argmax(0)])
            tmp_list = list()
            tmp_list.append(i)
            tmp_list.append(ytrue[i])
            tmp_list.extend(t_pred[i])
            score_list.append(tmp_list)
    
    return index_list, score_list

def make_logger(log_name, savedir='.logs/'):
    """Set up the logger for saving log file on the disk
    Args:
        cfg: configuration dict

    Return:
        logger: a logger for record essential information
    """
    import logging
    import os
    from logging.config import dictConfig
    import time

    logging_config = dict(
        version=1,
        formatters={'f_t': {
            'format': '\n %(asctime)s | %(levelname)s | %(name)s \t %(message)s'
        }},
        handlers={
            'stream_handler': {
                'class': 'logging.StreamHandler',
                'formatter': 'f_t',
                'level': logging.INFO
            },
            'file_handler': {
                'class': 'logging.FileHandler',
                'formatter': 'f_t',
                'level': logging.INFO,
                'filename': None,
            }
        },
        root={
            'handlers': ['stream_handler', 'file_handler'],
            'level': logging.DEBUG,
        },
    )
    # set up logger
    log_file = '{}.log'.format(log_name)
    # if folder not exist,create it
    if not os.path.exists(savedir):
        os.makedirs(savedir)
    log_file_path = os.path.join(savedir, log_file)

    logging_config['handlers']['file_handler']['filename'] = log_file_path

    open(log_file_path, 'w').close()  # Clear the content of logfile
    # get logger from dictConfig
    dictConfig(logging_config)

    logger = logging.getLogger()

    return logger