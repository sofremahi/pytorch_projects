

import torch
from torch import nn
class fasion_mnist_model(nn.Module):
   def __init__(self  , input_shape : int , hidden_units : int , output_shape : int):
          super().__init__()
          self.layer_stack =  nn.Sequential(
             nn.Flatten(),
             nn.Linear(in_features =input_shape , out_features = hidden_units ),
             nn.Linear(in_features =hidden_units , out_features =  output_shape) )       
   def forward(self , x):
      return self.layer_stack(x)  
class fasion_mnist_model_1(nn.Module):
   def __init__(self  , input_shape : int , hidden_units : int , output_shape : int):
          super().__init__()
          self.layer_stack =  nn.Sequential(
             nn.Flatten(),
             nn.Linear(in_features =input_shape , out_features = hidden_units ),
             nn.ReLU(),
             nn.Linear(in_features =hidden_units , out_features =  output_shape),
             nn.ReLU())       
   def forward(self , x):
      return self.layer_stack(x)  
#creating cnn Convolutional neural network   
class fashion_msnist_2(nn.Module):
   def __init__(self, input_shape: int, hidden_units: int, output_shape: int):
    super().__init__()
    self.conv_block_1 = nn.Sequential(
       
        nn.Conv2d(in_channels=input_shape, 
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.conv_block_2 = nn.Sequential(
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.Conv2d(in_channels=hidden_units,
                  out_channels=hidden_units,
                  kernel_size=3,
                  stride=1,
                  padding=1),
        nn.ReLU(),
        nn.MaxPool2d(kernel_size=2)
    )
    self.classifier = nn.Sequential(
        nn.Flatten(),
        nn.Linear(in_features=hidden_units*7*7, # trick of calculation
                  out_features=output_shape)
    )

   def forward(self, x):
    x = self.conv_block_1(x)
    # print(f"Output shape of conv_block_1: {x.shape}")
    x = self.conv_block_2(x) 
    # print(f"Output shape of conv_block_2: {x.shape}")
    x = self.classifier(x)
    # print(f"Output shape of classifier: {x.shape}")
    return x 