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

#our model is not learning the way    #lets create a tiny VGG with data augmentation #using schedular
#creating transform with trivialAugment


train_transform_trivial = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.TrivialAugmentWide(num_magnitude_bins=31),
    transforms.ToTensor()
])
test_transform_simple = transforms.Compose([
    transforms.Resize(size=(64,64)),
    transforms.ToTensor()
])
#turn to data sets
train_data_augmented = datasets.ImageFolder(root=train_dir,transform = train_transform_trivial)
test_data_simple = datasets.ImageFolder(root = test_dir , transform = test_transform_simple)
#turn into data loader 
import os
CPUs = os.cpu_count()
# torch.manual_seed(20)
train_dataloader_augmented = DataLoader(dataset = train_data_augmented ,
                                      batch_size = 32 ,
                                      num_workers = CPUs-CPUs,
                                      shuffle = True)
test_dataloader_simple = DataLoader(dataset = test_data_simple ,
                                     batch_size = 32 ,
                                     num_workers = 0 ,
                                     shuffle = False)
#training our tiny VGG with augmented data
model_1 = tiny_VGG(3 , 10 , len(train_data_augmented.classes))
loss_fn = torch.nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_1.parameters(),lr =0.001)
optimizer = torch.optim.Adam(model_1.parameters(),lr =0.001) 
# optimizer = torch.optim.SGD(model_1.parameters(), lr=0.01, momentum=0.9, weight_decay=1e-4)
# optimizer = torch.optim.Adam(model_1.parameters(), lr=0.001, weight_decay=1e-4)
# scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=0.01, gamma=0.1)
# scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.1, patience=5)
# scheduler = torch.optim.lr_scheduler.OneCycleLR(
#     optimizer, max_lr=0.01, steps_per_epoch=len(train_dataloader_augmented), epochs=50
# )
#train our model
train(model_1 ,train_dataloader_augmented,test_dataloader_simple,optimizer,loss_fn , epochs=20  )

#making results on a custom dataset not in train or test data 
#turning our custom image to tensors
import torchvision
image_path = "data/04-pizza-dad.jpeg"
custom_image_tensor = torchvision.io.read_image(image_path).type(torch.float32) /255
print(f"custom image shape is {custom_image_tensor.shape}")
#see our image
# from matplotlib import pyplot as plt
# plt.imshow(custom_image_tensor.permute(1,2,0))
# plt.show()

#transforming the custom image shape
custom_image_transform = transforms.Compose([transforms.Resize(size=(64,64))])
custom_tranformed_image = custom_image_transform(custom_image_tensor)
print(f"our transformed custom image shape is : {custom_tranformed_image.shape}")

#entering our custom image to our model
model_1.eval()
with torch.inference_mode():
    custom_image_logits = model_1(custom_tranformed_image.unsqueeze(0))
print(f"logits of the custom image from our model is {custom_image_logits}") 
custom_image_probs = torch.softmax(custom_image_logits , dim=1)
print(f"probabilities of the custom image from our model is {custom_image_probs}") 
custom_image_label = torch.argmax(torch.softmax(custom_image_logits , dim=1) , dim =1)
print(f"label of the custom image from our model is {train_dataset.classes[custom_image_label]} with index {custom_image_label}") 


#plot the prediction and the image
from helper import pred_and_plot_image
pred_and_plot_image(model_1,image_path,train_dataset.classes,custom_image_transform)
   
