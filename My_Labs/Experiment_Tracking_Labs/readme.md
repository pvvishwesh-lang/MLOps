# MLFlow Lab
## About
In this lab, I built a US Recession Prediction Model using FRED API Data, Pyspark and MLFlow

## Working
### ML Pipeline:
The dataset is downloaded from FRED API, after which we perform feature engineering and preprocessing on the data using Pyspark(which I've done on Databricks)
and downloaded the preprocessed dataset which has been saved as Silver Delta Table on my Databricks account. This silver delta table data is exported from databricks
and we perform further preprocessing such as creating 4 new columns(yield_spread,FedRates_3_months,Unrates_3_months,yield_spread_3_months) after which use VectorAssembler 
to further merge multiple input features into one output feature. Next, to avoid bias, we add a weight column that makes sure the lesser dominant features are given higher 
weight. We then train the model on a Logistic Regression model, evaluate it on AUC, Accuracy, Recall and Precision, all of which we then log to MLFLow UI.
Last, we store the model metrics in a folder, score the model to make sure it does not lose its accuracy etc and then use webbrowser package and subprocess package
to call localhost from within the python script.

## HOW TO REPLICATE
- Clone the github repo
- Navigate to MLFlow_Lab
- Run the following command: python MLFlow_Lab.py and wait for the run to be completed
- The python script will open a web tab with MLFLOW UI
- Navigate to Experiments->Recession Predictor->recession_predictor_v1 to visually see the metrics

## Analysis
The model performs exceptionally well, with an AUC of 0.9624819624819625,Accuracy of 0.8962264150943396,Recall of 0.8962264150943396 and
precision of 0.959643605870021. 

## Requirements
- Python
- Pyspark
- MLFlow


