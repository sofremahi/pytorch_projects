import torch
from torch import nn
from generate_chart import plot_predictions as plot
from linear import linear_expression as linear_module
#find ut what device is available 
device = "cuda" if torch.cuda.is_available else "cpu"
# creating train data and test data for our module
weight = 0.7
bias = 0.6
x_data = torch.arange(0 , 50)
y_data = x_data * weight + bias
split_rate =int(0.8 * len(x_data))
x_train = x_data[:split_rate]
y_train = y_data[:split_rate]
x_test = x_data[split_rate:]
y_test = y_data[split_rate:]
# creating the module with a manual seed 
torch.manual_seed(12)
model_0 = linear_module()
print(list(model_0.parameters()))
# giving test data to generate model before training
with torch.inference_mode():
  y_preds = model_0(x_test)
plot(x_train , y_train , x_test , y_test , predictions= y_preds)
#specify loss function and an optimizer 
loss_fn = nn.L1Loss()
#learning rate must be choosen with great visual of the model
optimizer = torch.optim.SGD(params=model_0.parameters(),lr=0.0001)

# loop throu and train model to get close to the actual model
#increasing the epochs will result in better and closer outcomes to the real data and model
epochs = 10000
for epoch in range(epochs):
  model_0.train()
  y_pred = model_0(x_train)
  loss = loss_fn(y_pred ,y_train)
  optimizer.zero_grad()
  loss.backward()
  optimizer.step()
  model_0.eval()
  with torch.inference_mode():
   test_pred = model_0(x_test)
   test_loss = loss_fn(test_pred , y_test)
  if epoch % 100 == 0:
      #check the loss and the test loss decresing as close to be 0.0000
    print(f"Epoch: {epoch} | Loss: {loss} | Test loss: {test_loss}")
    # Print out model state_dict()
    print(model_0.state_dict())
list(model_0.parameters()) 
with torch.inference_mode():
  y_preds = model_0(x_test)
plot(x_train , y_train , x_test , y_test , y_preds )
   
