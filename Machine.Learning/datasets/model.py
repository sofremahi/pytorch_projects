import torch
from torch import nn
class tiny_VGG(nn.Module):
    def __init__(self , input :int , hidden_layers:int , output:int):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = input , out_channels = hidden_layers ,
                       kernel_size = 3 ,
                       stride = 1 , 
                       padding=1
                       ) ,
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers , out_channels = hidden_layers ,
                       kernel_size = 3,
                       stride = 1 ,
                       padding = 1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels = hidden_layers, out_channels = hidden_layers ,
                       kernel_size = 3 ,
                       stride=1 ,
                       padding = 1),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers , out_channels = hidden_layers,
                       kernel_size = 3 ,
                       stride = 1 ,
                       padding =1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 )
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(in_features = hidden_layers * 16 * 16 ,
                      out_features = output))
    def forward(self , x):
         x = self.first_layer(x)
        #  print(f"shape is {x.shape}")
         x = self.second_layer(x)
        #  print(f"shape is {x.shape}")
         x = self.classifier(x)
         return x
     