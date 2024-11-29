import sys
import os
import torch
device = device=torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
#getting our needed data
from going_modular.predictions import pred_and_plot_image
from get_data import get_train_test_path
from transfer_efficientnet_p0 import get_class_names , get_trained_model, get_train_test_dataloader 
model = get_trained_model()
class_names = get_class_names()
train_dir , test_dir = get_train_test_path()
#get a random list of test image paths 
from pathlib import Path
import random
num_images = 5
test_image_path_list = list(Path(test_dir).glob("*/*.jpg"))
test_iamge_path_sample = random.sample(population=test_image_path_list , k=num_images)
from matplotlib import pyplot as plt
for image_path in test_iamge_path_sample:
    pred_and_plot_image(model,class_names,image_path,image_size=(224,224))
 
#creating a confusion matrix
#1 create a tesnor with all the right values of the data (y)
from tqdm import tqdm
train_dataloader , test_dataloader = get_train_test_dataloader()

y_preds = []
y_label=[]
model.to(device)
model.eval()
with torch.inference_mode():
      for x , y in tqdm(test_dataloader):
          x = x.to(device) 
          y = y.to(device)
          y_logits = model(x)
          print(f"logits shape {y_logits.shape}")
          y_pred = torch.softmax(y_logits.squeeze(),dim = 1).argmax(dim = 1)
          y_pred = torch.softmax(y_logits,dim = 1).argmax(dim = 1)
          print(f"squeeze logits shape {y_logits.squeeze().shape}")
          print(f"soft max shape {torch.softmax(y_logits.squeeze(),dim = 1).shape}")
          y_preds.append(y_pred.cpu())
          y_label.append(y.cpu())
y_pred_tensor = torch.cat(y_preds)  
y_label_tensor = torch.cat(y_label) 

from torchmetrics import ConfusionMatrix
from mlxtend.plotting import plot_confusion_matrix

# 2. Setup confusion instance and compare predictions to targets
confmat = ConfusionMatrix(task = "multiclass",num_classes=len(class_names))
confmat_tensor = confmat(preds = y_pred_tensor , target=y_label_tensor)
gig , ax = plot_confusion_matrix(conf_mat = confmat_tensor.numpy(),
                                 class_names = class_names,
                                 figsize = (10,7))
plt.show()