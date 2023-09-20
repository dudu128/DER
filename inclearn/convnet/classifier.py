import math

import torch
from torch.nn.parameter import Parameter
from torch.nn import functional as F
from torch.nn import Module
from inclearn.convnet.utils import cosine_similarity, stable_cosine_distance


class CosineClassifier(Module):
    def __init__(self, in_features, n_classes, sigma=True):
        super(CosineClassifier, self).__init__()
        self.in_features = in_features
        self.out_features = n_classes
        self.weight = Parameter(torch.Tensor(n_classes, in_features))
        if sigma:
            self.sigma = Parameter(torch.Tensor(1))
        else:
            self.register_parameter('sigma', None)
        self.reset_parameters()

    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma

    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out

# Cosine Classifier from PODNet
class PDCosineClassifier(Module):
    
    def __init__(self, Cosine_features, sigma=True):
        super(PDCosineClassifier, self).__init__()
        
        self.in_features = 0
        self.Cosine_features = Cosine_features
        self.n_classes = 0
        self.weight = nn.ParameterList([])
        self.device = device
        
        self.fc = None
    
    def reset_parameters(self):
        stdv = 1. / math.sqrt(self.weight.size(1))
        self.weight.data.uniform_(-stdv, stdv)
        if self.sigma is not None:
            self.sigma.data.fill_(1)  #for initializaiton of sigma
    
    
    def _add_new_classes(self, inc_dataset, in_features, n_classes, convnet):
        
        
        # FC add dimension
        fc = nn.Linear(in_features, self.Cosine_features, bias=True)
        nn.init.kaiming_normal_(self.fc, nonlinearity="linear")
        
        if self.reuse_oldfc:
            fc.weight.data[:self.Cosine_features, :in_features] = weight
        del self.fc
        self.fc = fc
        
        # Consine Classifier add dimension
        if self.imprint == False:
            new_weights = nn.Parameter(torch.zeros(n_classes, self.Cosine_features))
            nn.init.kaiming_normal_(new_weights, nonlinearity="linear")
            
            self.weight.append(new_weights)
            
        elif self.imprint == True:

            weights_norm = self.weights.data.norm(dim=1, keepdim=True)
            new_weights = []
            network = nn.Sequential(convnet, self.fc)

            for class_index in range(self.n_classes, self.n_classes+n_classes):
                _, _, loader = inc_dataset.get_custom_loader([class_index])
                features, _ = extract_features(network, loader)

                features_normalized = F.normalize(torch.from_numpy(features), p=2, dim=1)
                class_embeddings = torch.mean(features_normalized, dim=0)
                class_embeddings = F.normalize(class_embeddings, dim=0, p=2)

                new_weights.append(class_embeddings * avg_weights_norm)

            new_weights = torch.stack(new_weights)
            self.weight.append(nn.Parameter(new_wights))

        self.n_classes += n_classes
        
        return self
    
    def forward(self, input):
        out = F.linear(F.normalize(input, p=2, dim=1), F.normalize(self.weight, p=2, dim=1))
        if self.sigma is not None:
            out = self.sigma * out
        return out