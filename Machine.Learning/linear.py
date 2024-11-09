
import torch
from torch import nn
class linear_expression(nn.Module):
   def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1 , requires_grad=True, dtype = torch.float))
    self.bias = nn.Parameter(torch.randn(1 , requires_grad=True , dtype = torch.float))

   def forward(self , x : torch.Tensor) -> torch.Tensor:
      return self.weights * x + self.bias
