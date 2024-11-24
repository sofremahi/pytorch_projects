import torch
from torch import nn
class tiny_VGG(nn.Module):
    def __init__(self , input :int , hidden_layers:int , output:int):
        super().__init__()
        self.first_layer = nn.Sequential(
            nn.Conv2d(in_channels = input , out_channels = hidden_layers ,
                       kernel_size = 3 ,
                       stride = 1 , 
                       padding=0
                       ) ,
            #  nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers , out_channels = hidden_layers ,
                       kernel_size = 3,
                       stride = 1 ,
                       padding = 0),
            #  nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2)
        )
        self.second_layer = nn.Sequential(
            nn.Conv2d(in_channels = hidden_layers, out_channels = hidden_layers ,
                       kernel_size = 3 ,
                       stride=1 ,
                       padding = 0),
            #  nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.Conv2d(in_channels = hidden_layers , out_channels = hidden_layers,
                       kernel_size = 3 ,
                       stride = 1 ,
                       padding =0),
            #  nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size = 2 )
            )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            # nn.Dropout(0.5), 
            nn.Linear(in_features = hidden_layers * 13 * 13 ,
                      out_features = output))
    def forward(self , x):
         x = self.first_layer(x)
        #  print(f"shape is {x.shape}")
         x = self.second_layer(x)
        #  print(f"shape is {x.shape}")
         x = self.classifier(x)
         return x
     
     
class Tiny_VGG_v2(nn.Module):
    def __init__(self, input_channels: int, hidden_layers: int, output_classes: int):
        super().__init__()
        self.conv_block1 = nn.Sequential(
            nn.Conv2d(input_channels, hidden_layers, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.Conv2d(hidden_layers, hidden_layers, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.conv_block2 = nn.Sequential(
            nn.Conv2d(hidden_layers, hidden_layers * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers * 2),
            nn.ReLU(),
            nn.Conv2d(hidden_layers * 2, hidden_layers * 2, kernel_size=3, padding=1),
            nn.BatchNorm2d(hidden_layers * 2),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2)
        )
        self.classifier = nn.Sequential(
            nn.Flatten(),
            nn.Linear(hidden_layers * 2 * 16 * 16, 256),
            nn.ReLU(),
            nn.Dropout(p=0.5),
            nn.Linear(256, output_classes)
        )

    def forward(self, x):
        x = self.conv_block1(x)
        x = self.conv_block2(x)
        x = self.classifier(x)
        return x     