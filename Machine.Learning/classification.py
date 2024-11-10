import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import circle_model 


#check what device we are doing our machine learning computation
device = "cuda" if torch.cuda.is_available else "cpu"
#generating
n_samples =100
x , y = make_circles(n_samples , noise = 0.05 , random_state = 20)
print( f"first 10 of x : {x[:10]} and first 10 of y : {y[:10]}")
#creating data frames 
circles = pd.DataFrame({"X1" : x[:,0] ,
                        "X2" : x[:,1] ,
                        "Y" : y})
print(f"first 10 data frames : {circles.head(10)}")
#showing a circle chart data
circles.plot(kind='scatter', x='X1', y='X2', c='Y', colormap='viridis', title="Scatter plot of X1 vs X2")
#show tthe chart plot is needed 
### plt.show()

# as for machine learning and deep learning we need our data to be in form of tensors 
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#split data to train and test data sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state= 20)
#creating our specified model 
model_0 = circle_model().to(device)
#generating a model using nn.squential
model_0 = nn.sequential(
    nn.linear(in_features = 2 , out_features = 5),
    nn.Linear(in_features = 5 , out_features = 1)
).to(device)
with torch.infrence_mode():
 untrained_preds_test = model_0(x_test)
#setting los function and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGC(pararms = model_0.parameters() , lr=0.01)