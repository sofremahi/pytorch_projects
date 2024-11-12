
import torch
from torch import nn
class linear_expression(nn.Module):
   def __init__(self):
    super().__init__()
    self.weights = nn.Parameter(torch.randn(1 , requires_grad=True, dtype = torch.float))
    self.bias = nn.Parameter(torch.randn(1 , requires_grad=True , dtype = torch.float))

   def forward(self , x : torch.Tensor) -> torch.Tensor:
      return self.weights * x + self.bias 
   
   
class linear_expression_v2(nn.Module):
   def __init__(self):
      super().__init__()
      #creating linear layer
      self.linear_layer = nn.Linear(in_features=1,
                                  out_features=1)   
   
   def forward(self , x : torch.Tensor)-> torch.Tensor:
      return self.linear_layer(x)
   
   
class circle_model(nn.Module):
   def __init__(self):
      super().__init__()
      self.layer_1 = nn.Linear(in_features = 2 , out_features=5)
      self.layer_2 = nn.Linear(in_features = 5 , out_features =1)
   def forward(self , x ):
      return self.layer_2(self.layer_1(x))  


class circle_model_1(nn.Module):
   def __init__(self):
      super().__init__()
      self.layer_1= nn.Linear(in_features = 2 , out_features = 10)
      self.layer_2=nn.Linear(in_features = 10 , out_features = 10)
      self.layer_3=nn.Linear(in_features = 10 , out_features = 1)
   
   def forward(self , x ):
      return self.layer_3(self.layer_2(self.layer_1(x)))   
         