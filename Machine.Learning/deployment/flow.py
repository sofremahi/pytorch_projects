from pathlib import Path
import sys
import os
import torch
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))
from going_modular import data_setup , engine
pizza_steak_sushi_path = Path("Machine.Learning/data/")/"pizza_steak_sushi_20_percent"
print(f"path of our already downloaded data : {pizza_steak_sushi_path}")
train_dir = pizza_steak_sushi_path/"train"
test_dir = pizza_steak_sushi_path/"test"
from pre_trained import create_effnetb2_model

#create efficient net b2 

effnet_b2 , effnet_b2_transforms = create_effnetb2_model(num_classes=3)
#create data loaders
effnet_train_dataloader , effnet_test_dataloader , class_names = data_setup.create_dataloaders(train_dir,test_dir,effnet_b2_transforms,batch_size=32 , num_workers=0) 
# optimizer
optimizer = torch.optim.Adam(params=effnet_b2.parameters(),
                             lr=1e-3)
# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Set seeds for reproducibility and train the model
effnetb2_results = engine.train(model=effnet_b2,
                                train_dataloader=effnet_train_dataloader,
                                test_dataloader=effnet_test_dataloader,
                                epochs=5,
                                optimizer=optimizer,
                                loss_fn=loss_fn,
                                device=device)
from helper_functions import plot_loss_curves
from matplotlib import pyplot as plt
plot_loss_curves(effnetb2_results)
# plt.show()

#create vit 16

from pre_trained import create_vit_model
vit, vit_transforms = create_vit_model(num_classes=len(class_names))
# Setup ViT DataLoaders
train_dataloader_vit, test_dataloader_vit, class_names = data_setup.create_dataloaders(train_dir=train_dir,
                                                                                       test_dir=test_dir,
                                                                                       transform=vit_transforms,
                                                                                       batch_size=32 , num_workers=0)
#train our vit pre trained model
#  optimizer
optimizer = torch.optim.Adam(params=vit.parameters(),
                             lr=1e-3)
# loss function
loss_fn = torch.nn.CrossEntropyLoss()

# Train ViT
# vit_results = engine.train(model=vit,
#                            train_dataloader=train_dataloader_vit,
#                            test_dataloader=test_dataloader_vit,
#                            epochs=5,
#                            optimizer=optimizer,
#                            loss_fn=loss_fn,
#                            device=device)
# plot_loss_curves(vit_results)
# plt.show()
#create a list of test data paths
test_data_paths = list(Path(test_dir).glob("*/*.jpg"))
from going_modular import utils
effnetb2_test_pred_dicts = utils.pred_and_store(paths=test_data_paths,
                                          model=effnet_b2,
                                          transform=effnet_b2_transforms,
                                          class_names=class_names,
                                          device="cpu") 
import pandas as pd
effnetb2_test_pred_df = pd.DataFrame(effnetb2_test_pred_dicts)
print(effnetb2_test_pred_df.head())
#                                             image_path class_name  pred_prob pred_class  time_for_pred  correct
# 0  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9596      pizza         0.1396     True
# 1  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.4836      pizza         0.0984     True
# 2  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9711      pizza         0.1415     True
# 3  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.6020      pizza         0.0751     True
# 4  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.6754      pizza         0.0691     True
# vit_test_pred_dicts = utils.pred_and_store(paths=test_data_paths,
#                                      model=vit,
#                                      transform=vit_transforms,
#                                      class_names=class_names,
#                                      device="cpu")
import pandas as pd
# vit_test_pred_df = pd.DataFrame(vit_test_pred_dicts)
# print(vit_test_pred_df.head())
#                                           image_path class_name  pred_prob pred_class  time_for_pred  correct
# 0  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9985      pizza         0.1872     True
# 1  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9954      pizza         0.1892     True
# 2  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9987      pizza         0.1480     True
# 2  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9987      pizza         0.1480     True
# 2  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9987      pizza         0.1480     True
# 3  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9900      pizza         0.1547     True
# 4  Machine.Learning\data\pizza_steak_sushi_20_per...      pizza     0.9734      pizza         0.1501     True

def return_effnet_b2_needs():
    return effnet_b2 , effnet_b2_transforms
def return_ViT():
    return vit

from PIL import Image
import random
random_image_path = random.sample(test_data_paths, k=1)[0]
image = Image.open(random_image_path)

from util import predict
pred_dict, pred_time = predict(img=image , class_names=class_names , model=effnet_b2 , transform=effnet_b2_transforms)

# Create a list of example inputs to our Gradio demo
example_list = [[str(filepath)] for filepath in random.sample(test_data_paths, k=3)]


#processing the full flow
import gradio as gr
# Create title, description and article strings
title = "FoodVision Mini 🍕🥩🍣"
description = "An EfficientNetB2 feature extractor computer vision model to classify images of food as pizza, steak or sushi."
article = "[PyTorch Model Deployment]"

# Create the Gradio demo
demo = gr.Interface(fn=lambda img: predict(img, class_names, effnet_b2, effnet_b2_transforms), 
                    inputs=gr.Image(type="pil"), 
                    outputs=[gr.Label(num_top_classes=3, label="Predictions"), 
                             gr.Number(label="Prediction time (s)")], 
                    examples=example_list, 
                    title=title,
                    description=description,
                    article=article)

# Launch the demo
demo.launch(debug=False,
            share=True) 