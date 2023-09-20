import numpy as np
import random
import time
import math
import os
import json
from copy import deepcopy
from scipy.spatial.distance import cdist

import torch
from torch.nn import DataParallel
from torch.nn import functional as F

from inclearn.convnet import network
from inclearn.models.base import IncrementalLearner
from inclearn.tools import factory, utils
from inclearn.tools.metrics import ClassErrorMeter
from inclearn.tools.memory import MemorySize
from inclearn.tools.scheduler import GradualWarmupScheduler
from inclearn.convnet.utils import extract_features, update_classes_mean, finetune_last_layer

# Constants
EPSILON = 1e-8


class IncModel(IncrementalLearner):
    def __init__(self, cfg, trial_i, _run, ex, tensorboard, inc_dataset):
        super().__init__()
        self._cfg = cfg
        self._device = cfg['device']
        self._ex = ex
        self._run = _run  # the sacred _run object.

        # Data
        self._inc_dataset = inc_dataset
        self._n_classes = 0
        self._trial_i = trial_i  # which class order is used

        # Optimizer paras
        self._opt_name = cfg["optimizer"]
        self._warmup = cfg['warmup']
        self._lr = cfg["lr"]
        self._weight_decay = cfg["weight_decay"]
        self._n_epochs = cfg["epochs"]
        self._scheduling = cfg["scheduling"]
        self._lr_decay = cfg["lr_decay"]

        # Classifier Learning Stage
        self._decouple = cfg["decouple"]

        # Logging
        self._tensorboard = tensorboard
        if f"trial{self._trial_i}" not in self._run.info:
            self._run.info[f"trial{self._trial_i}"] = {}
        self._val_per_n_epoch = cfg["val_per_n_epoch"]

        # Model
        self._der = cfg['der']  # Whether to expand the representation
        self._network = network.BasicNet(
            cfg["convnet"],
            cfg=cfg,
            nf=cfg["channel"],
            device=self._device,
            use_bias=cfg["use_bias"],
            dataset=cfg["dataset"],
        )
        self._parallel_network = DataParallel(self._network)
        self._train_head = cfg["train_head"]
        self._infer_head = cfg["infer_head"]
        self._old_model = None

        # Learning
        self._temperature = cfg["temperature"]
        self._distillation = cfg["distillation"]

        # Memory
        self._memory_size = MemorySize(cfg["mem_size_mode"], inc_dataset, cfg["memory_size"],
                                       cfg["fixed_memory_per_cls"], cfg["openset"])
        self._herding_matrix = []
        self._coreset_strategy = cfg["coreset_strategy"]

        if self._cfg["save_ckpt"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            if not os.path.exists(save_path):
                os.mkdir(save_path)
            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                if not os.path.exists(save_path):
                    os.mkdir(save_path)
        
        self._gce = cfg["gce"]
        self._gce_topk = cfg["gce_topk"]
        self._openset = cfg["openset"]
        self._deep_fc = cfg["deep_fc"]
        self.rotation =  cfg["rotation"]
        self.jigsaw = cfg["jigsaw"]
        self.jigsaw_patches = cfg["jigsaw_patches"]
        self.jigsaw_loc = cfg["loc"]
        self.RCM = cfg["RCM"]
        self._aux_open = cfg["aux_open"]
        self.mixup = cfg["mixup"]
        self.mixup_old = cfg["mixup_old"]
        self.mixup_old_others = cfg["mixup_old_others"]
        self.mixup_old_p = cfg["mixup_old_p"]
        self.memory_loader = None
        self.memory_iter = None
        self._gen_status = True

    def eval(self):
        self._parallel_network.eval()

    def train(self):
        if self._der:
            self._parallel_network.train()
            self._parallel_network.module.convnets[-1].train()
            if self._task >= 1:
                for i in range(self._task):
                    self._parallel_network.module.convnets[i].eval()
        else:
            self._parallel_network.train()

    def _before_task(self, taski, inc_dataset):
        self._ex.logger.info(f"Begin step {taski}")

        # Update Task info
        self._task = taski
        self._n_classes += self._task_size

        # Memory
        self._memory_size.update_n_classes(self._n_classes)
        self._memory_size.update_memory_per_cls(self._network, self._n_classes, self._task_size)
        self._ex.logger.info("Now {} examplars per class.".format(self._memory_per_class))

        self._network.add_classes(self._task_size)
        self._network.task_size = self._task_size
        self.set_optimizer()

    def set_optimizer(self, lr=None):
        if lr is None:
            lr = self._lr

        if self._cfg["dynamic_weight_decay"]:
            # used in BiC official implementation
            weight_decay = self._weight_decay * self._cfg["task_max"] / (self._task + 1)
        else:
            weight_decay = self._weight_decay
        self._ex.logger.info("Step {} weight decay {:.5f}".format(self._task, weight_decay))

        if self._der and self._task > 0:
            for i in range(self._task):
                for p in self._parallel_network.module.convnets[i].parameters():
                    p.requires_grad = False

        self._optimizer = factory.get_optimizer(filter(lambda p: p.requires_grad, self._network.parameters()),
                                                self._opt_name, lr, weight_decay)

        if "cos" in self._cfg["scheduler"]:
            self._scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(self._optimizer, self._n_epochs)
        else:
            self._scheduler = torch.optim.lr_scheduler.MultiStepLR(self._optimizer,
                                                                   self._scheduling,
                                                                   gamma=self._lr_decay)

        if self._warmup:
            print("warmup")
            self._warmup_scheduler = GradualWarmupScheduler(self._optimizer,
                                                            multiplier=1,
                                                            total_epoch=self._cfg['warmup_epochs'],
                                                            after_scheduler=self._scheduler)

    def _train_task(self, train_loader, val_loader):
        self._ex.logger.info(f"nb {len(train_loader.dataset)}")

        topk = 5 if self._n_classes > 5 else self._task_size
        accu = ClassErrorMeter(accuracy=True, topk=[1, topk])
        train_new_accu = ClassErrorMeter(accuracy=True)
        train_old_accu = ClassErrorMeter(accuracy=True)

        # utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "Initial trainset")
        # utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
        #                            self._increments, "Initial trainset")
        
        self._optimizer.zero_grad()
        self._optimizer.step()
        
        # if True:
        #     self.mixup_old_p = len(self._inc_dataset.targets_cur) / len(self._inc_dataset.targets_inc)
        
        for epoch in range(self._n_epochs):
            _loss, _loss_aux, _loss_jig = 0.0, 0.0, 0.0
            accu.reset()
            train_new_accu.reset()
            train_old_accu.reset()
            if self._warmup:
                self._warmup_scheduler.step()
                if epoch == self._cfg['warmup_epochs']:
                    if self._deep_fc == False:
                        self._network.classifier.reset_parameters()
                    else:
                        self._network.classifier[0].reset_parameters()
                        self._network.classifier[1].reset_parameters()
                    if self._cfg['use_aux_cls']:
                        self._network.aux_classifier.reset_parameters()
            for i, (inputs, targets) in enumerate(train_loader, start=1):
                self.train()
                self._optimizer.zero_grad()
                old_classes = targets < (self._n_classes - self._task_size)
                new_classes = targets >= (self._n_classes - self._task_size)
                
                # print(old_classes)
                
                loss_ce, loss_aux, loss_rot, loss_jig = self._forward_loss(
                    inputs,
                    targets,
                    old_classes,
                    new_classes,
                    accu=accu,
                    new_accu=train_new_accu,
                    old_accu=train_old_accu,
                )

                if self._cfg["use_aux_cls"] and self._task > 0:
                    loss = loss_ce + loss_aux
                else:
                    loss = loss_ce
                   
                if self.rotation:
                    loss += 0.1 * loss_rot
                
                if self.jigsaw:
                    loss += loss_jig

                if not utils.check_loss(loss):
                    import pdb
                    pdb.set_trace()

                loss.backward()
                self._optimizer.step()

                if self._cfg["postprocessor"]["enable"]:
                    if self._cfg["postprocessor"]["type"].lower() == "wa":
                        for p in self._network.classifier.parameters():
                            p.data.clamp_(0.0)

                _loss += loss_ce
                _loss_aux += loss_aux
                _loss_jig += loss_jig
            _loss = _loss.item()
            _loss_aux = _loss_aux.item()
            if self.jigsaw:
                _loss_jig = _loss_jig.item()
            if not self._warmup:
                self._scheduler.step()
            self._ex.logger.info(
                "Task {}/{}, Epoch {}/{} => Clf loss: {} Aux loss: {}, Jig loss: {} Train Accu: {}, Train@5 Acc: {}, old acc:{}".
                format(
                    self._task + 1,
                    self._n_tasks,
                    epoch + 1,
                    self._n_epochs,
                    round(_loss / i, 3),
                    round(_loss_aux / i, 3),
                    round(_loss_jig / i, 3),
                    round(accu.value()[0], 3),
                    round(accu.value()[1], 3),
                    round(train_old_accu.value()[0], 3),
                ))

            if self._val_per_n_epoch > 0 and epoch % self._val_per_n_epoch == 0:
                self.validate(val_loader)

            if self._gen_status:
                status = dict()
                status['task'] = self._task
                status['epoch'] = epoch
                status['loss'] = _loss
                status['acc'] = accu.value()[0]
                status['status'] = "Training"
                status['completed'] = False
                with open("./status.json", 'w') as f:
                    json.dump(status, f)
                    
        # For the large-scale dataset, we manage the data in the shared memory.
        self._inc_dataset.shared_data_inc = train_loader.dataset.share_memory
        if self._deep_fc == False:
            utils.display_weight_norm(self._ex.logger, self._parallel_network, self._increments, "After training")
            utils.display_feature_norm(self._ex.logger, self._parallel_network, train_loader, self._n_classes,
                                       self._increments, "Trainset")
        self._run.info[f"trial{self._trial_i}"][f"task{self._task}_train_accu"] = round(accu.value()[0], 3)

    def _forward_loss(self, inputs, targets, old_classes, new_classes, accu=None, new_accu=None, old_accu=None):
        
        inputs, targets = inputs.to(self._device), targets.to(self._device)
        
        lam = 0
        if self.mixup_old:
            
            old_batch_size = int(inputs.shape[0] * self.mixup_old_p)
            num_img = int(inputs.shape[0] - old_batch_size)
            
            if self.mixup_old_others:
                aux_data = np.concatenate((self._inc_dataset.data_memory, self._inc_dataset.data_others))
                aux_targets = np.concatenate((self._inc_dataset.targets_memory, self._inc_dataset.targets_others))
            else:
                aux_data = self._inc_dataset.data_memory
                aux_targets = self._inc_dataset.targets_memory
            
            mem_idx = np.random.choice(len(aux_targets), old_batch_size, replace=False)
            mem_inputs = aux_data[mem_idx]
            mem_targets = aux_targets[mem_idx]
            
            memory_loader = iter(self._inc_dataset._get_loader(mem_inputs, mem_targets, mode="train", batch_size=old_batch_size))
            
            alpha = self._cfg["mixup_a"]
            lam = np.random.beta(alpha, alpha)
            
            index = torch.randperm(inputs.size(0))[:num_img]
            
            mem_inputs, mem_targets = next(memory_loader)
            mem_inputs, mem_targets = mem_inputs.to(self._device), mem_targets.to(self._device)
            
            mixup_img = torch.cat((inputs[index], mem_inputs))
            mixup_targets = torch.cat((targets[index], mem_targets))
            mixup_old_classes = torch.cat((old_classes[index], torch.ones(old_batch_size).bool()))
            mixup_new_classes = torch.cat((new_classes[index], torch.zeros(old_batch_size).bool()))
            
            inputs = lam * inputs + (1 - lam) * mixup_img
            targets_a, targets_b = targets, mixup_targets
            targets = torch.stack((targets_a, targets_b))
            old_classes = torch.stack((old_classes, mixup_old_classes))
            new_classes = torch.stack((new_classes, mixup_new_classes))

        elif self.mixup:
            alpha = self._cfg["mixup_a"]
            lam = np.random.beta(alpha, alpha)
            index = torch.randperm(inputs.size(0))
            
            inputs = lam*inputs + (1-lam)*inputs[index, :]
            targets_a, targets_b = targets, targets[index]
            targets = torch.stack((targets_a, targets_b))
            old_classes = torch.stack((old_classes, old_classes[index]))
            new_classes = torch.stack((new_classes, new_classes[index]))
        
        if self.jigsaw:
            inputs1, jigsaw_law = self.jigsaw_generator(inputs, self.jigsaw_patches, self.RCM)
            jigsaw = torch.cat((torch.zeros(inputs.shape[0], dtype=torch.int64), torch.ones(inputs.shape[0], dtype=torch.int64)), 0).clone().detach().to(self._device)
            inputs = torch.cat((inputs, torch.tensor(inputs1).to(self._device)), 0)
            targets = torch.cat((targets, targets), -1).to(self._device)
            old_classes = torch.cat((old_classes, old_classes), -1)
            new_classes = torch.cat((new_classes, new_classes), -1)
        
        outputs = self._parallel_network(inputs)
        if accu is not None:
            if self.mixup:
                accu.add(outputs['logit'], targets[0])
            else:
                accu.add(outputs['logit'], targets)
        
        loss_ce, loss_aux = self._compute_loss(inputs, targets, outputs, old_classes, new_classes, lam)
        
        rot_loss = 0
        if self.rotation:
            rotated_inputs = [inputs]
            angles = [torch.zeros(len(inputs))]
            
            for ang in range(1, 4):
                rotated_inputs.append(inputs.rot90(ang, [2, 3]))
                angles.append(torch.ones(len(inputs)) * ang)
            
            rotated_inputs = torch.cat(rotated_inputs)
            angles = torch.cat(angles).long().to(self._device)
            
            outputs = self._parallel_network(rotated_inputs, rotation=True)
            rot_loss = F.cross_entropy(outputs["rotations"], angles)
        
        jig_loss = 0
        if self.jigsaw:
            adv_loss = F.cross_entropy(outputs["jigsaw_adv"], jigsaw)
            patches = self.jigsaw_patches ** 2
            swap_law = []
            swap_law1 = [(i-self.jigsaw_patches)/patches for i in range(patches)]
            swap_law2 = []
            for i in range(len(jigsaw_law)):
                swap_law.append(swap_law1)
                swap_law2.append([(j-self.jigsaw_patches)/patches for j in jigsaw_law[i]])
            swap_law.extend(swap_law2)
            swap_law = torch.tensor(swap_law).to(self._device)
            loc_loss = F.l1_loss(outputs["mask"], swap_law)
            
            jig_loss = adv_loss + loc_loss * self.jigsaw_loc
            
        
        return loss_ce, loss_aux, rot_loss, jig_loss

    def _compute_loss(self, inputs, targets, outputs, old_classes, new_classes, lam):
        
        if self.mixup:
            loss = lam * F.cross_entropy(outputs['logit'], targets[0]) + (1-lam) * F.cross_entropy(outputs['logit'], targets[1])
        else:
            loss = F.cross_entropy(outputs['logit'], targets)
        
        if self._gce:
            gce_topk = self._gce_topk
            x = outputs['logit'].cpu()
            y = targets.cpu()
            x1 = x.clone().cpu()
            x1[range(x1.size(0)), y] = -float("Inf")
            x_gt = x[range(x.size(0)), y].unsqueeze(1)
            x_topk = torch.topk(x1, gce_topk, dim=1)[0]  # 15 Negative classes to focus on, its a hyperparameter
            x_new = torch.cat([x_gt, x_topk], dim=1)
            loss += F.cross_entropy(x_new.to(self._device), torch.zeros(x_new.size(0)).long().to(self._device))
        
        if self._aux_open:
            open_targets = targets.clone()
            open_targets = [0 if i == 0 else 1 for i in open_targets].to(self._device)
            loss += F.cross_entropy(outputs['open_logit'], open_targets)
        

        if outputs['aux_logit'] is not None:
            aux_targets = targets.clone()
            
            if self._cfg["aux_n+1"]:
                aux_targets[old_classes] = 0
                aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 1
            elif self._cfg["aux_n+2"]:
                others = targets == 0
                aux_targets[old_classes] = 1
                aux_targets[others] = 0
                aux_targets[new_classes] -= sum(self._inc_dataset.increments[:self._task]) - 2
                
            if self.mixup:
                aux_loss = lam * F.cross_entropy(outputs['logit'], aux_targets[0]) + (1-lam) * F.cross_entropy(outputs['logit'], aux_targets[1])
            else:
                aux_loss = F.cross_entropy(outputs['aux_logit'], aux_targets)
        else:
            aux_loss = torch.zeros([1]).cuda()
        
        return loss, aux_loss

    def _after_task(self, taski, inc_dataset, val_loader):
        network = deepcopy(self._parallel_network)
        network.eval()
        self._ex.logger.info("save model")
        if self._cfg["save_ckpt"] and taski >= self._cfg["start_task"]:
            save_path = os.path.join(os.getcwd(), "ckpts")
            torch.save(network.cpu().state_dict(), "{}/step{}.ckpt".format(save_path, self._task))

        if (self._cfg["decouple"]['enable'] and taski > 0):
            if self._cfg["decouple"]["fullset"]:
                # train_loader = inc_dataset._get_loader(inc_dataset.data_inc, inc_dataset.targets_inc, mode="train")
                train_loader = inc_dataset._get_loader(inc_dataset.data_trained, inc_dataset.targets_trained, mode="train")
            else:
                train_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                       inc_dataset.targets_inc,
                                                       mode="balanced_train")
            print(len(inc_dataset.targets_inc))
            # finetuning
            if self._cfg["decouple"]['reset_fc']:
                if self._deep_fc == True:
                    self._parallel_network.module.classifier[0].reset_parameters()
                    self._parallel_network.module.classifier[1].reset_parameters()
                else:
                    self._parallel_network.module.classifier.reset_parameters()
            finetune_last_layer(self._ex.logger,
                                self._parallel_network,
                                train_loader,
                                self._n_classes,
                                nepoch=self._decouple["epochs"],
                                lr=self._decouple["lr"],
                                scheduling=self._decouple["scheduling"],
                                lr_decay=self._decouple["lr_decay"],
                                weight_decay=self._decouple["weight_decay"],
                                loss_type="ce",
                                temperature=self._decouple["temperature"],
                                scheduler_type=self._decouple["type"],
                                test_loader=val_loader,
                                mixup=self._decouple["mixup"],
                                alpha=self._decouple["alpha"],
                                optimizer_type=self._decouple["optim"],
                                )
            network = deepcopy(self._parallel_network)
            if self._cfg["save_ckpt"]:
                save_path = os.path.join(os.getcwd(), "ckpts")
                torch.save(network.cpu().state_dict(), "{}/decouple_step{}.ckpt".format(save_path, self._task))

        if self._cfg["postprocessor"]["enable"]:
            self._update_postprocessor(inc_dataset)

        if self._cfg["infer_head"] == 'NCM':
            self._ex.logger.info("compute prototype")
            self.update_prototype()

        if self._memory_size.memsize != 0:
            self._ex.logger.info("build memory")
            self.build_exemplars(inc_dataset, self._coreset_strategy)

            if self._cfg["save_mem"]:
                save_path = os.path.join(os.getcwd(), "ckpts/mem")
                memory = {
                    'x': inc_dataset.data_memory,
                    'y': inc_dataset.targets_memory,
                    'herding': self._herding_matrix
                }
                if not os.path.exists(save_path):
                    os.makedirs(save_path)
                if not (os.path.exists(f"{save_path}/mem_step{self._task}.ckpt") and self._cfg['load_mem']):
                    torch.save(memory, "{}/mem_step{}.ckpt".format(save_path, self._task))
                    self._ex.logger.info(f"Save step{self._task} memory!")

        self._parallel_network.eval()
        self._old_model = deepcopy(self._parallel_network)
        self._old_model.module.freeze()
        del self._inc_dataset.shared_data_inc
        self._inc_dataset.shared_data_inc = None

    def _eval_task(self, data_loader):
        if self._infer_head == "softmax":
            ypred, ytrue, features = self._compute_accuracy_by_netout(data_loader)
        elif self._infer_head == "NCM":
            ypred, ytrue = self._compute_accuracy_by_ncm(data_loader)
        else:
            raise ValueError()

        return ypred, ytrue, features

    def _compute_accuracy_by_netout(self, data_loader):
        preds, targets = [], []
        features = None
        imgs = None
        self._parallel_network.eval()
        with torch.no_grad():
            for i, (inputs, lbls) in enumerate(data_loader):
                
                inputs = inputs.to(self._device, non_blocking=True)
                _outputs = self._parallel_network(inputs)
                _preds = _outputs['logit']
                _features = _outputs['feature']
                
                if self._cfg["postprocessor"]["enable"] and self._task > 0:
                    _preds = self._network.postprocessor.post_process(_preds, self._task_size)
                    
                preds.append(_preds.detach().cpu().numpy())
                targets.append(lbls.long().cpu().numpy())
                
                feat = _features.cpu().numpy()
                if features is not None:
                    features = np.concatenate((features, feat))
                else:
                    features = feat
                
        preds = np.concatenate(preds, axis=0)
        targets = np.concatenate(targets, axis=0)
        return preds, targets, features

    def _compute_accuracy_by_ncm(self, loader):
        features, targets_ = extract_features(self._parallel_network, loader)
        targets = np.zeros((targets_.shape[0], self._n_classes), np.float32)
        targets[range(len(targets_)), targets_.astype("int32")] = 1.0

        class_means = (self._class_means.T / (np.linalg.norm(self._class_means.T, axis=0) + EPSILON)).T

        features = (features.T / (np.linalg.norm(features.T, axis=0) + EPSILON)).T
        # Compute score for iCaRL
        sqd = cdist(class_means, features, "sqeuclidean")
        score_icarl = (-sqd).T
        return score_icarl[:, :self._n_classes], targets_

    def _update_postprocessor(self, inc_dataset):
        if self._cfg["postprocessor"]["type"].lower() == "bic":
            if self._cfg["postprocessor"]["disalign_resample"] is True:
                bic_loader = inc_dataset._get_loader(inc_dataset.data_inc,
                                                     inc_dataset.targets_inc,
                                                     mode="train",
                                                     resample='disalign_resample')
            else:
                xdata, ydata = inc_dataset._select(inc_dataset.data_train,
                                                   inc_dataset.targets_train,
                                                   low_range=0,
                                                   high_range=self._n_classes)
                bic_loader = inc_dataset._get_loader(xdata, ydata, shuffle=True, mode='train')
            bic_loss = None
            self._network.postprocessor.reset(n_classes=self._n_classes)
            self._network.postprocessor.update(self._ex.logger,
                                               self._task_size,
                                               self._parallel_network,
                                               bic_loader,
                                               loss_criterion=bic_loss)
        elif self._cfg["postprocessor"]["type"].lower() == "wa":
            self._ex.logger.info("Post processor wa update !")
            self._network.postprocessor.update(self._network.classifier, self._task_size)

    def update_prototype(self):
        if hasattr(self._inc_dataset, 'shared_data_inc'):
            shared_data_inc = self._inc_dataset.shared_data_inc
        else:
            shared_data_inc = None
        self._class_means = update_classes_mean(self._parallel_network,
                                                self._inc_dataset,
                                                self._n_classes,
                                                self._task_size,
                                                share_memory=self._inc_dataset.shared_data_inc,
                                                metric='None')

    def build_exemplars(self, inc_dataset, coreset_strategy):
        save_path = os.path.join(os.getcwd(), f"ckpts/mem/mem_step{self._task}.ckpt")
        if self._cfg["load_mem"] and os.path.exists(save_path):
            memory_states = torch.load(save_path)
            self._inc_dataset.data_memory = memory_states['x']
            self._inc_dataset.targets_memory = memory_states['y']
            self._herding_matrix = memory_states['herding']
            self._ex.logger.info(f"Load saved step{self._task} memory!")
            return

        if coreset_strategy == "random":
            from inclearn.tools.memory import random_selection

            self._inc_dataset.data_memory, self._inc_dataset.targets_memory = random_selection(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._ex.logger,
                inc_dataset,
                self._memory_per_class,
            )
        elif coreset_strategy == "iCaRL":
            from inclearn.tools.memory import herding
            data_inc = self._inc_dataset.shared_data_inc if self._inc_dataset.shared_data_inc is not None else self._inc_dataset.data_inc
            self._inc_dataset.data_memory, self._inc_dataset.targets_memory, self._herding_matrix = herding(
                self._n_classes,
                self._task_size,
                self._parallel_network,
                self._herding_matrix,
                inc_dataset,
                data_inc,
                self._memory_per_class,
                self._ex.logger,
                self._openset
            )
        else:
            raise ValueError()

    def validate(self, data_loader):
        if self._infer_head == 'NCM':
            self.update_prototype()
        ypred, ytrue, featrues = self._eval_task(data_loader)
        
        test_acc_stats = utils.compute_accuracy(ypred, ytrue, increments=self._increments, n_classes=self._n_classes)
        self._ex.logger.info(f"test top1acc:{test_acc_stats['top1']}")
        
        
    def jigsaw_generator(self, images, n, RCM):

        images = images.cpu().numpy()
        block_size = images.shape[2] // n
        rounds = n ** 2
        jigsaws = []
        permutations = []
        for im in images:
            
            l = np.arange(0, n ** 2, 1, dtype=int)
            permut = []
            
            if RCM == False:
                random.shuffle(l)
                permut = l
            else:
                tmpx = []
                tmpy = []
                count_x = 0
                count_y = 0
                k = 1
                RAN = 2
                for i in range(rounds):
                    tmpx.append(l[i])
                    count_x += 1
                    if len(tmpx) >= k:
                        tmp = tmpx[count_x - RAN:count_x]
                        random.shuffle(tmp)
                        tmpx[count_x - RAN:count_x] = tmp
                    if count_x == n:
                        tmpy.append(tmpx)
                        count_x = 0
                        count_y += 1
                        tmpx = []
                    if len(tmpy) >= k:
                        tmp2 = tmpy[count_y - RAN:count_y]
                        random.shuffle(tmp2)
                        tmpy[count_y - RAN:count_y] = tmp2
                for line in tmpy:
                    permut.extend(line)

            jig = deepcopy(im)
            tmp = deepcopy(im)

            for i in range(rounds):
                o_x, o_y = permut[i] % n, permut[i] // n
                t_x, t_y = i % n , i // n

                jig[..., t_x * block_size:(t_x + 1) * block_size, t_y * block_size:(t_y + 1) * block_size] = tmp[..., o_x * block_size:(o_x + 1) * block_size, o_y * block_size:(o_y + 1) * block_size]
                
            jigsaws.append(jig)
            permutations.append(permut)
        return np.array(jigsaws), permutations