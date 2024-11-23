import torch
from become_1with_data import train_dir , test_dir
from torchvision import transforms , datasets
from torch.utils.data import DataLoader
#creating our transform
simple_transform = transforms.Compose([
    transforms.Resize(size=(64 ,64)),
    transforms.ToTensor()
])
#creating datasets
train_dataset = datasets.ImageFolder(root=train_dir,
                                     transform = simple_transform)
test_dataset = datasets.ImageFolder(root=test_dir,
                                    transform = simple_transform)
#creating data loaders
train_simple_dataloader = DataLoader(dataset = train_dataset,
                                     batch_size = 32 ,
                                     num_workers = 0,
                                     shuffle=True)
test_simple_dataloader = DataLoader(dataset = test_dataset,
                                     batch_size = 32 ,
                                     num_workers = 0,
                                     shuffle=False)
x , y = next(iter(train_simple_dataloader))
from model import tiny_VGG
model_0 = tiny_VGG(input=3 , hidden_layers=10 , output=len(train_dataset.classes))
print(f"passing x of shape : {x.shape} into our model with 3 color channel input , 3 classes pizza,steak,sushi output")
logits = model_0(x)
print(f"32 packs of logits from model in a 32,3 tensor shape : {logits.shape}") 
# from torchinfo import summary
# print(summary(model_0  ,  input_size=[1,3,64,64]))

#training out model
import tqdm
from helper import train
#choosing our loss function and optimizer 
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters() , lr=0.01)
train(model_0,train_simple_dataloader,test_simple_dataloader,optimizer,loss_fn)

#our model is not learning the way    #lets create a tiny VGG with data augmentation