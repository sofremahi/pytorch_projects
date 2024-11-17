#import torch
import torch
from torch import nn
#import torch vision needs
import torchvision
from torchvision import datasets
from torchvision import transforms
from torchvision.transforms import ToTensor
import matplotlib.pyplot as plt

#check version 
print(torch.__version__)
print(torchvision.__version__)

#get a dataset
train_data = datasets.FashionMNIST(
    root = "data" ,# where we are going to downlaod the data
    train = True, # we want the training data set ?
    download = True , # do we want to download it ?
    transform = torchvision.transforms.ToTensor(), # how to transform the data
    target_transform = None , # do we want to transform the lables/targets ?
    
)
test_data = datasets.FashionMNIST(
    root = "data" ,
    train = False,
    download = True ,
    transform = torchvision.transforms.ToTensor(), 
    target_transform = None , 
)
image , label = train_data[0]
print(image , label)
print(len(test_data) , len(train_data))
#data classes
class_names = train_data.classes
print(class_names)
#data classes representing index
al_labels = train_data.class_to_idx
print(al_labels)
print(f"{image.shape}color channel , height , width , {label}")
#for ploting we need to get rid of extrac dimension or put the color channles at last HWC
print(image.squeeze().shape)
#visualizing the data
# plt.imshow(image.squeeze())
# plt.title(label)
# plt.imshow(image.squeeze() )
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
#ploting more images
# torch.manual_seed(22)
# fig = plt.figure(figsize=(9 , 9))
# rows , cols = 4 , 4 
# for i in range(1 , (rows*cols)+1):
    #chossing a random index with random seed flavored
#  random_idx = torch.randint(0  , len(train_data) , size=[1] ).item()
 #choose an image based on our random index
#  img , label = train_data[random_idx]
#  fig.add_subplot(rows , cols, i)
#  plt.imshow(img.squeeze() , cmap="gray")
#  plt.title(class_names[label])
#  plt.axis(False)
# plt.show() 
#turn data sets into pytorch iterables and created batches to process one batch by batch
#better to train the data (mini_batches) batch_size = 32 

from torch.utils.data import DataLoader
BATCH_SIZE = 32
#turn into interables
train_data_loader = DataLoader(dataset = train_data , 
                               batch_size = BATCH_SIZE ,
                               shuffle = True)
test_data_loader = DataLoader(dataset = test_data , 
                               batch_size = BATCH_SIZE ,
                               shuffle = False)
#whats inside the data loaders
train_features_batch , train_labels_batch = next(iter(train_data_loader))
print(train_features_batch.shape  , train_labels_batch.shape)
#show a sample 
# torch.manual_seed(22)
# random_idx = torch.randint(0 , len(train_features_batch) , size =[1] ).item()
# image , label = train_features_batch[random_idx]  , train_labels_batch[random_idx] 
# plt.imshow(image.squeeze() , cmap="gray")
# plt.title(class_names[label])
# plt.axis(False)
# plt.show()
#creating our first Computing model :: creating a flatter layer 
flatten_model = nn.Flatten()
#get a single sample
x = train_features_batch[0]
#flatten the sample
output = flatten_model(x)
print(f"{x.shape} color channels , height , width {output.shape} color channel , height*width")
#creating out model
from model import fasion_mnist_model
model_0 =  fasion_mnist_model(input_shape= 28*28 ,
                              hidden_units= 8 ,
                              output_shape=len(class_names))
#import accuracy metric
from helper_functions import accuracy_fn
#choosing loss function and the optimizer
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters(),lr=0.1)
#create our timeout function
from timeit import default_timer as timer
def print_time(start : float , end :float , device : torch.device = None):
    total_time = end - start 
    print(f"total time on device {device} is {total_time:.3f}")
    return total_time
# from tqdm.auto import tqdm
torch.manual_seed(22) 
start_time = timer()
#start training model 
epochs = 6
for epoch in range(epochs):
    print(f"Epoch: {epoch}\n------")
    train_loss = 0 
    #add a loop to go through out the training batches
    for batch , (x,y) in enumerate(train_data_loader):
        model_0.train()
        y_pred = model_0(x)
        loss = loss_fn(y_pred , y )
        train_loss += loss
        #optimizer 
        optimizer.zero_grad()
        #loss backwards
        loss.backward()
        #optimizer step
        optimizer.step()
        if batch % 400 == 0 :
            print(f"looked at {batch  * len(x)}/{(len(train_data_loader.dataset))} samples")
    train_loss /= len(train_data_loader)
    
    #testing 
    test_loss , acc = 0 , 0
    model_0.eval() 
    with torch.inference_mode():
     for x , y in test_data_loader:
        test_pred = model_0(x)    
        #calculate the loss
        test_loss =+ loss_fn(test_pred , y)  
        #calculate the accuracy 
        acc = accuracy_fn(y_true=y , y_pred= test_pred.argmax(dim=1) ) 
     #calculate the test loss        
     test_loss /= len(test_data_loader)
     acc /= len(test_data_loader) 
     # Print out what's happening
    print(f"\nTrain loss: {train_loss:.4f} | Test loss: {test_loss:.4f}, Test acc: {acc:.4f}")

end_time = timer()     
total_time = print_time(start=start_time , end=end_time)