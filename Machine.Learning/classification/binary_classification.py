import sklearn
from sklearn.datasets import make_circles
from sklearn.model_selection import train_test_split
import pandas as pd
import matplotlib.pyplot as plt
import torch
from torch import nn
from model import  circle_model , circle_model_1 , circle_model_2
from help_functions import accuracy_fn
from helper_functions import plot_predictions, plot_decision_boundary
from generate_chart import plot_predictions_circle_boundary

#check what device we are doing our machine learning computation
device = "cuda" if torch.cuda.is_available else "cpu"
#generating
n_samples =10000
x , y = make_circles(n_samples , noise = 0.05 , random_state = 20)
print( f"first 10 of x : {x[:10]} and first 10 of y : {y[:10]}")
#creating data frames 
circles = pd.DataFrame({"X1" : x[:,0] ,
                        "X2" : x[:,1] ,
                        "Y" : y})
print(f"first 10 data frames : {circles.head(10)}")
#showing a circle chart data
plt.scatter(x[:, 0], x[:, 1], c=y, cmap=plt.cm.RdYlBu);
#create the plot by data frames created with pandas
# circles.plot(kind='scatter', x='X1', y='X2', c='Y', colormap='viridis', title="Scatter plot of X1 vs X2")
#show tthe chart plot if needed 
### plt.show()

# as for machine learning and deep learning we need our data to be in form of tensors 
x = torch.from_numpy(x).type(torch.float)
y = torch.from_numpy(y).type(torch.float)
#split data to train and test data sets
x_train , x_test , y_train , y_test = train_test_split(x , y , test_size= 0.2 , random_state= 20)
#creating our specified model 
model_0 = circle_model()
#generating a model using nn.squential
model_0 = nn.Sequential(
    nn.Linear(in_features = 2 , out_features = 5),
    nn.Linear(in_features = 5 , out_features = 1)
)
with torch.inference_mode():
 untrained_preds_test = model_0(x_test)
#setting los function and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_0.parameters() , lr=0.1)

#train the model
model_0.eval()
with torch.inference_mode():
 y_logits = model_0(x_test)[:5]
print(y_logits)
#use sigmoid to turn our model logits into prediction probabilities
y_pred_probs = torch.sigmoid(y_logits) 
print(y_pred_probs)
#lets rounf the sigmoids and find the prediction lables
y_pred = torch.round(y_pred_probs)
print(y_pred)
# in  full :: logits --> pred probs --> pred lables
y_pred_lables = torch.round(torch.sigmoid(model_0(x_test)[:5])) 
print(y_pred.squeeze()==y_pred_lables.squeeze())

#building a training and testing loop
#creating our custom seed
torch.manual_seed(20)
torch.cuda.manual_seed(20)
epochs = 200
#loop 
for epoch in range(epochs):
    #training
    model_0.train()
    #forward pass
    y_logits = model_0(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    #initializing loss function 
    # if loss_fn is initialized with nn.BCELoss it wants probabilities as train data
    # loss = loss_fn(torch.sigmoid(y_logits) , y_train)
    # is loss_fn is initialzied with nn.BCEWithLogitsLoss it wants logits and train data
    loss = loss_fn(y_logits ,y_train)
    #calculate the loss accuracy
    acc = accuracy_fn(y_True  = y_train , 
                      y_preds= y_pred)
    #optimize zero grad
    optimizer.zero_grad
    #loss backward
    loss.backward()
    #optimizer step
    optimizer.step()
    
    #time for testing
    model_0.eval()
    test_logits = model_0(x_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    #calculate the test loss
    test_loss = loss_fn(test_logits , y_test)
    test_acc = accuracy_fn(y_True= y_test , y_preds=test_pred)
    
    ## print whats happening
    if epoch % 100 == 0 :
        print(f"in epoch : {epoch} and the loss : {loss:.5f} and the accuracy : {acc:.2f}")
        print(f"test loss : {test_loss:.5f} and the test accuracy : {test_acc:.2f}")

#plot decision boundary of our model with helper_functions.py 
# plt.figure(figsize=(12,6))
# plt.subplot(1,2,1)  
# plt.title("train")
# plot_decision_boundary(model_0 , x_train , y_train) 
# plt.subplot(1,2,2)    
# plt.title("test")
# plot_decision_boundary(model_0 , x_test , y_test)
# plt.show()

#try training our model much sufficient and with accessible methods 
#higher layers - higher hidden layers - more epochs - better loss function/optimizer choosing
model_1 = circle_model_1()
print(model_1)
#setting los function and optimizer 
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD(params = model_1.parameters() , lr=0.1)
#setting manual seed
torch.manual_seed(20)
torch.cuda.manual_seed(20)
#going throughout training and testin loop
epochs = 1000
for epoch in range(epochs):
    #training model
    model_1.train()
    y_logits = model_1(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    #calculate the loss
    loss = loss_fn(y_logits , y_train)
    acc = accuracy_fn(y_True=y_train , y_preds=y_pred)
    #zero grad
    optimizer.zero_grad()
    # back 
    loss.backward()
    # step optimizer
    optimizer.step()
    
    #testin model
    model_1.eval()
    test_logits = model_1(x_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits , y_test)
    test_acc = accuracy_fn(y_True=y_test , y_preds=test_pred)
    if epoch % 100 == 0:
            print(f"in epoch : {epoch} and the loss : {loss:.5f} and the accuracy : {acc:.2f}")
            print(f"test loss : {test_loss:.5f} and the test accuracy : {test_acc:.2f}")
plot_predictions_circle_boundary(model_1 , x_train , y_train , x_test , y_test)
#show the plot
# plt.show()

#train our model to be a none linear traning model 
model_2 = circle_model_2()
#choosing loss function and optimizer
loss_fn = nn.BCEWithLogitsLoss()
optimizer = torch.optim.SGD( model_2.parameters() , lr = 0.1)
#training loop for our model
epochs = 1000
for epoch in range(epochs):
    #training model
    model_1.train()
    y_logits = model_2(x_train).squeeze()
    y_pred = torch.round(torch.sigmoid(y_logits))
    #calculate the loss
    loss = loss_fn(y_logits , y_train)
    acc = accuracy_fn(y_True=y_train , y_preds=y_pred)
    #zero grad
    optimizer.zero_grad()
    # back 
    loss.backward()
    # step optimizer
    optimizer.step()
    
    #testin model
    model_1.eval()
    test_logits = model_2(x_test).squeeze()
    test_pred = torch.round(torch.sigmoid(test_logits))
    test_loss = loss_fn(test_logits , y_test)
    test_acc = accuracy_fn(y_True=y_test , y_preds=test_pred)
    if epoch % 100 == 0:
            print(f"in epoch : {epoch} and the loss : {loss:.5f} and the accuracy : {acc:.2f}")
            print(f"test loss : {test_loss:.5f} and the test accuracy : {test_acc:.2f}")
            
plot_predictions_circle_boundary(model_2 , x_train , y_train , x_test , y_test)
plt.show()