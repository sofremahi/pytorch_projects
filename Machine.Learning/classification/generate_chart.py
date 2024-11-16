import matplotlib.pyplot as plt
from helper_functions import plot_predictions, plot_decision_boundary

def plot_predictions(train_data,
                     train_labels,
                     test_data,
                     test_labels,
                     predictions):
  """
  Plots training data, test data and compares predictions.
  """
  plt.figure(figsize=(10, 7))

  plt.scatter(train_data, train_labels, c="b", s=4, label="Training data")

  plt.scatter(test_data, test_labels, c="g", s=4, label="Testing data")


  if predictions is not None:

    plt.scatter(test_data, predictions, c="r", s=4, label="Predictions")

  plt.legend(prop={"size": 14});
      # Show the plot
  plt.show()
  
  
def plot_predictions_circle_boundary(model , x_train , y_train , 
                                     x_test , y_test)  :
  #plot decision boundary of our model with helper_functions.py 
  plt.figure(figsize=(12,6))
  plt.subplot(1,2,1)  
  plt.title("train")
  plot_decision_boundary(model , x_train , y_train) 
  plt.subplot(1,2,2)    
  plt.title("test")
  plot_decision_boundary(model , x_test , y_test)