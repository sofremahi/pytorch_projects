import gradio
import torch
import torchvision
#Creating function for gradio input - process - output
from timeit import default_timer as timer
from typing import Tuple , Dict 
def predict(img , class_names, model: torch.nn.Module,
                   transform: torchvision.transforms, )-> Tuple[Dict , float]:
    #start timer 
    start_time = timer()

    img = transform(img).unsqueeze(dim=0) # transform image and add a batch to it
    model.eval()
    with torch. inference_mode():
        pred_probs = torch.softmax(model(img) , dim=1)
        #create the dict for prediction probabilities
        pred_labels_and_probs = {class_names[i]: float(pred_probs[0][i]) for i in range(len(class_names))}
        #end timer
    end_time = timer()
    
    return pred_labels_and_probs , round(end_time-start_time)    
   
    