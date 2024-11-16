import torch
from torch import nn
import sklearn
from sklearn.datasets import make_blobs
from sklearn.model_selection import train_test_split
import matplotlib.pyplot as plt
from models.model import blob_model
from help_functions import accuracy_fn
from generate_chart import plot_predictions_circle_boundary

#specify our static values 
NUM_CALSSES = 4
NUM_FEATURES = 2
RANDOM_SEED = 20
#create our data set ready
x_blob , y_blob = make_blobs(n_samples=1000 , n_features=NUM_FEATURES , 
                               centers=NUM_CALSSES ,
                               cluster_std=1.7 ,#a little shake up
                               random_state =RANDOM_SEED
                               )
#turn data to tensors 
x_blob = torch.from_numpy(x_blob).type(torch.float)
y_blob = torch.from_numpy(y_blob).type(torch.LongTensor)
#Spliting our data to train data and test data
x_blob_train , x_blob_test , y_blob_train , y_blob_test = train_test_split(
    x_blob , y_blob , test_size= 0.3 , random_state= RANDOM_SEED
)
#plot the data visualize
plt.figure(figsize=(10, 7))
plt.scatter(x_blob[:, 0], x_blob[:, 1], c=y_blob, cmap=plt.cm.RdYlBu);
#show the plot 
# plt.show()
#instantiating  out model
print(x_blob_train.shape, y_blob_train.shape)
model_0 = blob_model(input_features = 2 , 
                     output_features = 4 ,
                     hidden_units = 8)

#seeing our model
model_0.eval()
with torch.inference_mode():
    y_logits = model_0(x_blob_test)
    #these are the logits
    #in future we need to convert our logits to prediction probabilities 
    #and then for our accuracy we need to convert them to predection lables torch.round() 
    
    
#logits    
print(y_logits[:5])    
#the real data
print(y_blob_test[:5])
#prediction probabilities
y_pred_probs = torch.softmax(y_logits , dim =1)
print(y_pred_probs[:5])
#the sum of every panel of prediction probabilities are 1 as the actual probability
#and the highest probability value and index (our model is not trained)
print(torch.sum(y_pred_probs[0]) , torch.max(y_pred_probs[0]) , torch.argmax(y_pred_probs[0]))


#creating loss function and optimizer good enough for our multi class classificaiton problem
loss_fn = nn.CrossEntropyLoss()
optimizer = torch.optim.SGD(model_0.parameters() , lr=0.1)

#setting upt manual seed 
torch.manual_seed(20)
torch.cuda.manual_seed(20)
#creating our training and testing loop
epochs = 10000
#loop
for epoch in range(epochs):
    #training
    model_0.train()
    logits = model_0(x_blob_train)
    y_pred = torch.softmax(logits, dim=1).argmax(dim=1)
    
    loss = loss_fn( logits ,y_blob_train)
    acc = accuracy_fn(y_True=y_blob_train , y_preds= y_pred)
    
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()
    #testing
    model_0.eval()
    with torch.inference_mode():
        test_logits = model_0(x_blob_test)
        test_pred = torch.softmax(test_logits, dim=1).argmax(dim=1)
        test_loss = loss_fn(test_logits , y_blob_test)
        test_acc = accuracy_fn(y_True= y_blob_test , y_preds=test_pred)
        
    if epoch % 100 == 0 :
        print(f"epoch {epoch} , the loss is :{loss} and the accuracy is :{acc}")    
        print(f" the test_loss is :{test_loss} and the test_accuracy is :{test_acc}")    
plot_predictions_circle_boundary(model_0 , x_blob_train ,
                                 y_blob_train , x_blob_test , y_blob_test)
plt.show()   
 





