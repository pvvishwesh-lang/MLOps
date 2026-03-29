# Tensorboard Lab
## About
In this lab, I built a tensorflow based neural network model that takes the Boston Housing dataset from Keras.datasets and logs everything for visualization.

## Working
### DATASET:
The dataset is from keras.datasets.bostonhousing, which contains 13 different attributes of houses from different locations in Boston suburbs in late 1970's.
The target variable is the median price is $. Though there is a warning about potential racial bias by the authors of the dataset, I've used this dataset since
it had small number of features and could be trained quickly. 

### WHAT THE CODE DOES:
In this lab, we first load the boston housing dataset from keras, use mean and std to normalize the input features. We then create model from keras.sequential
with dense layers options set and compile and fit the model on the dataset. We save the logs for this in the logs folder to use for the visualization using 
tensorboard. We use the webbrowser python package to open the localhost port on webbrowser since by default the tensorboard tries to open it within the jupyter 
notebook cell.

## HOW TO REPLICATE
- Clone the dataset from the github repo
- Navigate to My_Labs/Tensorflow_Labs.
- Run the notebook cells.
- Once all the cells finish their runs, a new terminal in the webbrowser is opened with the dashboard
- Navigate through the dashboard to visualize various metrics such as loss etc

## Requirements
- Python 3.10+  
- TensorFlow 2.x  
- Jupyter Notebook / JupyterLab
