from fastai.vision import *
from .nb_Tesis import *

class NeuralIluminant(torch.nn.Module):
    def __init__(self):
        super(NeuralIluminant, self).__init__()
        self.iluminant = torch.nn.Parameter(torch.tensor([1.0,1.0,1.0]).cuda())
    def forward(self, x):
        return x * self.iluminant[:,None,None]
    
    
