import random
from PIL import Image
from matplotlib import pyplot as plt
def plot_transformed_images(image_paths , transform , n=3 , seed =None):
    if seed:
        random.seed(seed)
    random_image_paths = random.sample(image_paths, k=n)   
    for image_path in random_image_paths:
        with Image.open(image_path) as f:
         fig , ax = plt.subplots(nrows=1 , ncols=2)
         ax[0].imshow(f)
         ax[0].set_title(f"Original\nSize: {f.size}")
         ax[0].axis(False)
         transformed_image = transform(f).permute(1, 2, 0) #  (C, H, W) -> (H, W, C)
         ax[1].imshow(transformed_image)
         ax[1].set_title(f"Transformed\nShape: {transformed_image.shape}")
         ax[1].axis("off")

        fig.suptitle(f"Class: {image_path.parent.stem}", fontsize=16)
    plt.show()    
        
from typing import Tuple , Dict , List  
import os       
def find_classes_in_directory(directory : str)->Tuple[List[str] , Dict[str , int]]:
    #get the classes sorted
    classes = sorted([entry.name for entry in list(os.scandir(directory))])
    if not classes:
        raise FileNotFoundError(f"didnt fine any file at path {directory}")
    class_to_idx = {class_name : i for i , class_name in enumerate(classes)}
    
    return classes , class_to_idx 
from torch.utils.data import Dataset
import pathlib
import torch
class ImageFolderCustom(Dataset):
#initialize our custom dataset
 def __init__(self , directory  : str , transform :None ):
     #set up paths 
     self.paths = list(pathlib.Path(directory).glob("*/*.jpg")) # its going to be our test or train dir
     #set up transform
     self.transfom = transform 
     self.classes, self.class_to_idx = find_classes_in_directory(directory)
 def load_image(self , index:int)->Image.Image:
     "opens an iamge file path"
     image_path = self.paths[index]
     return Image.open(image_path)
 def __len__(self):
     return len(self.paths)  
 def __getitem__(self ,index: int )-> Tuple[torch.Tensor , int] :
     """return a sample of the data ,  data and label (x,y) """  
     imge = self.load_image(index)  
     class_name = self.paths[index].parent.name # -> data_folder/class_name/image.jpg
     class_to_idx = self.class_to_idx[class_name] 
     if self.transfom:
      return self.transfom(imge) , class_to_idx

def display_random_images(dataset : torch.utils.data.Dataset , 
                          classes: list[str] = None ,
                          n : int = 10  ,
                          display_shape = True ,
                          seed :int = None):
    if seed :
        random.seed(seed)
    random_samples_idx = random.sample(range(len(dataset)) , k = n) 
    plt.figure(figsize=(16,8))
    #loop through out random indexes and plot them 
    for i , target_sample in enumerate(random_samples_idx)   :
        target_image , target_label = dataset[target_sample][0],dataset[target_sample][1]
        target_image_permuted = target_image.permute(1,2,0)# (C,H,W)->(H,W,C)
        plt.subplot(1 , n ,i+1)
        plt.imshow(target_image_permuted)
        plt.axis("off")
        if classes:
            title =  f"class {classes[target_label]}"
            if display_shape:
                title = title + f"\nshape : {target_image_permuted.shape}"
        plt.title(title) 
    plt.show()    
    
    
    from torch import nn       
def train_step(model :torch.nn.Module,
               dataloader:torch.utils.data.DataLoader,
               loss_fn : torch.nn.Module ,
               optimizer : torch.optim.Optimizer,
               device = None) :
    model.train() 
    #setup train loss and accuracy
    train_loss , acc = 0,0
    #loop 
    for batch , (x,y) in enumerate(dataloader):
       y_logits = model(x)
       loss = loss_fn(y_logits , y)
       train_loss += loss.item()
       optimizer.zero_grad()
       loss.backward()
       optimizer.step() 
       y_pred = torch.argmax(torch.softmax(y_logits , dim=1), dim=1)
       acc += (y_pred==y).sum().item() / len(y_pred)
    train_loss /= len(dataloader)
    acc /= len(dataloader)   
    # print(f"train loss : {train_loss} and train acc : {acc}")  
    return train_loss , acc


def test_step(model:torch.nn.Module,
              dataloader:torch.utils.data.DataLoader,
              loss_fn = torch.nn.Module,
              device =None):
    model.eval()
    #set up test loss and test acc
    test_loss , test_acc = 0 , 0
    with torch.inference_mode():
     for batch ,(x,y) in enumerate(dataloader):
        y_logits = model(x)
        test_loss += loss_fn(y_logits , y).item()
        test_pred = torch.argmax(torch.softmax(y_logits , dim=1),dim=1)
        test_acc += (test_pred==y).sum().item() / len(test_pred)
     test_loss /= len(dataloader)
     test_acc /= len(dataloader)  
    #  print(f"test loss : {test_loss} and test acc : {test_acc}")  
     return test_loss , test_acc  
 

from tqdm import tqdm

def train(model: torch.nn.Module,
          train_dataloader: torch.utils.data.DataLoader,
          test_dataloader : torch.utils.data.DataLoader,
          optimizer : torch.optim.Optimizer,
          loss_fn: torch.nn.Module = torch.nn.CrossEntropyLoss(),
          epochs : int = 5 , 
          device=None):
  #create a DICT
  results = {"train_loss": [],
             "train_acc": [],
             "test_loss": [],
             "test_acc": []}
  #looping throu train and test 
  for epoch in tqdm(range(epochs)):
    train_loss, train_acc = train_step(model,
                                       train_dataloader,
                                       loss_fn,
                                       optimizer,
                                       device)
    test_loss, test_acc = test_step(model,
                                    test_dataloader,
                                    loss_fn,
                                    device)
#see whats happening
    print(f"Epoch: {epoch} | Train loss: {train_loss:.4f} | Train acc: {train_acc:.4f} | Test loss: {test_loss:.4f} | Test acc: {test_acc:.4f}")
    #set dictionary results
    results["train_loss"].append(train_loss)
    results["train_acc"].append(train_acc)
    results["test_loss"].append(test_loss)
    results["test_acc"].append(test_acc)

  return results
      