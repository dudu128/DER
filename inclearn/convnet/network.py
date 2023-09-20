import copy
import pdb

import torch
from torch import nn
import torch.nn.functional as F
from torch.nn import DataParallel

from inclearn.tools import factory
from inclearn.convnet.imbalance import BiC, WA
from inclearn.convnet.classifier import CosineClassifier
from inclearn.convnet.utils import extract_features


class BasicNet(nn.Module):
    def __init__(
        self,
        convnet_type,
        cfg,
        nf=64,
        use_bias=False,
        init="kaiming",
        device=None,
        dataset="cifar100",
    ):
        super(BasicNet, self).__init__()
        self.nf = nf
        self.init = init
        self.convnet_type = convnet_type
        self.dataset = dataset
        self.start_class = cfg['start_class']
        self.weight_normalization = cfg['weight_normalization']
        self.remove_last_relu = True if self.weight_normalization else False
        self.use_bias = use_bias if not self.weight_normalization else False
        self.der = cfg['der']
        self.aux_nplus1 = cfg['aux_n+1']
        self.aux_nplus2 = cfg['aux_n+2']
        self.reuse_oldfc = cfg['reuse_oldfc']
        self.myclassifier = False
        self.sci = cfg['sci']
        self.deep_fc = cfg['deep_fc']
        self.rotation = cfg['rotation']
        self.jigsaw = cfg['jigsaw']
        self.pretrain_model = cfg['pretrain_model']
        self.jigsaw_stride = 4 if cfg['jigsaw_patches'] == 4 else int(14/cfg['jigsaw_patches'])
        print(self.jigsaw_stride)
        self.aux_open = cfg['aux_open']
        self.load_old_model = cfg['load_old_model']

        if self.der:
            print("Enable dynamical reprensetation expansion!")
            self.convnets = nn.ModuleList()
            self.convnets.append(
                factory.get_convnet(convnet_type,
                                    nf=nf,
                                    dataset=dataset,
                                    start_class=self.start_class,
                                    remove_last_relu=self.remove_last_relu,
                                    sci=self.sci,
                                    pretrain_model=self.pretrain_model))
            self.out_dim = self.convnets[0].out_dim
        else:
            self.convnet = factory.get_convnet(convnet_type,
                                               nf=nf,
                                               dataset=dataset,
                                               remove_last_relu=self.remove_last_relu,
                                               sci=self.sci,
                                               pretrain_model=self.pretrain_model)
            self.out_dim = self.convnet.out_dim
        self.classifier = None
        self.aux_classifier = None
        self.rot_classifier = None
        self.adv_classifier = None
        self.open_classifier = None
        self.Convmask = None
        self.avgpool2 = nn.AvgPool2d(1, stride=1).to(device)

        self.n_classes = 0
        self.ntask = 0
        self.device = device

        if cfg['postprocessor']['enable']:
            if cfg['postprocessor']['type'].lower() == "bic":
                self.postprocessor = BiC(cfg['postprocessor']["lr"], cfg['postprocessor']["scheduling"],
                                         cfg['postprocessor']["lr_decay_factor"], cfg['postprocessor']["weight_decay"],
                                         cfg['postprocessor']["batch_size"], cfg['postprocessor']["epochs"])
            elif cfg['postprocessor']['type'].lower() == "wa":
                self.postprocessor = WA()
        else:
            self.postprocessor = None

        print(self.device)
        self.to(self.device)

    def forward(self, x, rotation=False):
        if self.classifier is None:
            raise Exception("Add some classes before training.")

        if self.der:
            features = []
            for convnet in self.convnets:
                x1, x2 = convnet(x)
                features.append(x1)
            # features = [convnet(x) for convnet in self.convnets]
            features = torch.cat(features, 1)
        else:
            features, x1 = self.convnet(x)

        logits = self.classifier(features)

        aux_logits = self.aux_classifier(features[:, -self.out_dim:]) if features.shape[1] > self.out_dim else None
        
        if rotation:
            rot_logits = self.rot_classifier(features[:, -self.out_dim:])
        else:
            rot_logits = None

        if self.jigsaw:
            jigsaw_adv = self.adv_classifier(features[:, -self.out_dim:])

            mask = self.Convmask(x2)
            mask = self.avgpool2(mask)
            mask = torch.tanh(mask)
            mask = mask.view(mask.size(0), -1)
        else:
            jigsaw_adv = None
            mask = None
        
        if self.aux_open:
            open_logits = self.open_classifier(features[:, -self.out_dim:])
        else:
            open_logits = None

        # return features
        return {'feature': features, 'logit': logits, 'aux_logit': aux_logits, 'rotations': rot_logits, 'jigsaw_adv': jigsaw_adv, 'mask': mask, 'open_logit': open_logits}

    @property
    def features_dim(self):
        if self.der:
            return self.out_dim * len(self.convnets)
        else:
            return self.out_dim

    def freeze(self):
        for param in self.parameters():
            param.requires_grad = False
        self.eval()
        return self

    def copy(self):
        return copy.deepcopy(self)

    def add_classes(self, n_classes):
        self.ntask += 1

        if self.der:
            self._add_classes_multi_fc(n_classes)
        else:
            self._add_classes_single_fc(n_classes)

        self.n_classes += n_classes

    def _add_classes_multi_fc(self, n_classes):
        if self.ntask > 1:
            new_clf = factory.get_convnet(self.convnet_type,
                                          nf=self.nf,
                                          dataset=self.dataset,
                                          start_class=self.start_class,
                                          remove_last_relu=self.remove_last_relu,
                                          sci=self.sci,
                                          pretrain_model=self.pretrain_model).to(self.device)            
            new_clf.load_state_dict(self.convnets[-1].state_dict())
            self.convnets.append(new_clf)

        if self.classifier is not None and self.deep_fc == False:
            weight = copy.deepcopy(self.classifier.weight.data)
        elif self.classifier is not None and self.deep_fc == True:
            weight_0 = copy.deepcopy(self.classifier[0].weight.data)
            weight_1 = copy.deepcopy(self.classifier[1].weight.data)

        fc = self._gen_classifier(self.out_dim * len(self.convnets), self.n_classes + n_classes, deep_fc=self.deep_fc)

        if self.classifier is not None and self.reuse_oldfc and self.deep_fc == False:
            fc.weight.data[:self.n_classes, :self.out_dim * (len(self.convnets) - 1)] = weight
        elif self.classifier is not None and self.deep_fc == True:
            fc[0].weight.data[:512, :self.out_dim * (len(self.convnets) - 1)] = weight_0
            fc[1].weight.data[:self.n_classes, :512] = weight_1
        del self.classifier
        self.classifier = fc

        if self.aux_nplus1:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 1)
        elif self.aux_nplus2:
            aux_fc = self._gen_classifier(self.out_dim, n_classes + 2)
        else:
            aux_fc = self._gen_classifier(self.out_dim, self.n_classes + n_classes)
        del self.aux_classifier
        self.aux_classifier = aux_fc
        
        if self.rotation:
            rot_fc = self._gen_classifier(self.out_dim, 4)
            del self.rot_classifier
            self.rot_classifier = rot_fc
        
        if self.jigsaw:
            jig_fc = self._gen_classifier(self.out_dim, 2)
            del self.adv_classifier
            self.adv_classifier = jig_fc
            
            convmask = nn.Conv2d(self.out_dim, 1, 1, stride=self.jigsaw_stride, padding=0, bias=True).to(self.device)
            del self.Convmask
            self.Convmask = convmask

        if self.aux_open:
            open_fc = self._gen_classifier(self.out_dim, 2)
            del self.open_classifier
            self.open_classifier = open_fc

        if self.load_old_model:
            load_model = DataParallel(self.copy())
            load_model.load_state_dict(torch.load(f'./weight/step{self.ntask-1}.ckpt'), strict=False)
            # print(load_model)
            self.convnets[-1].load_state_dict(load_model.module.convnets[-1].state_dict())

    def _add_classes_single_fc(self, n_classes):
        if self.classifier is not None:
            weight = copy.deepcopy(self.classifier.weight.data)
            if self.use_bias:
                bias = copy.deepcopy(self.classifier.bias.data)

        classifier = self._gen_classifier(self.features_dim, self.n_classes + n_classes)

        if self.classifier is not None and self.reuse_oldfc:
            classifier.weight.data[:self.n_classes] = weight
            if self.use_bias:
                classifier.bias.data[:self.n_classes] = bias

        del self.classifier
        self.classifier = classifier

    def _gen_classifier(self, in_features, n_classes, deep_fc=False):
        if self.weight_normalization:
            if deep_fc:
                classifier = nn.Sequential(
                    nn.Linear(in_features, 512, bias=self.use_bias),
                    CosineClassifier(512, n_classes)
                ).to(self.device)
            else:
                classifier = CosineClassifier(in_features, n_classes).to(self.device)
        else:
            if deep_fc:
                classifier = nn.Sequential(
                    nn.Linear(in_features, 512, bias=self.use_bias),
                    nn.Linear(512, n_classes, bias=self.use_bias)
                ).to(self.device)
            else:
                classifier = nn.Linear(in_features, n_classes, bias=self.use_bias).to(self.device)
                
            if self.init == "kaiming":
                if deep_fc:
                    nn.init.kaiming_normal_(classifier[0].weight, nonlinearity="linear")
                    nn.init.kaiming_normal_(classifier[1].weight, nonlinearity="linear")
                else:
                    nn.init.kaiming_normal_(classifier.weight, nonlinearity="linear")
            if self.use_bias:
                nn.init.constant_(classifier.bias, 0.0)

        return classifier
    
