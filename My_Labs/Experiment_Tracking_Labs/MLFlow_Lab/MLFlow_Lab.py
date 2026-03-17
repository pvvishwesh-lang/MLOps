import warnings
from pyspark.sql import SparkSession
from pyspark.sql import functions as F
from pyspark.sql import Window
from pyspark.ml.feature import VectorAssembler
from pyspark.ml.classification import LogisticRegression,LogisticRegressionModel
from pyspark.ml.evaluation import BinaryClassificationEvaluator,MulticlassClassificationEvaluator
from pyspark.ml.functions import vector_to_array
import mlflow
import os
import logging
import subprocess
import webbrowser
import time

subprocess.run(['pip','install','mlflow','pyspark==3.5.0'])

mlflow.set_experiment("recession_predictor")
logging.getLogger("py4j").setLevel(logging.WARNING)
warnings.filterwarnings(action='ignore')

os.makedirs('Datasets',exist_ok=True)
os.makedirs('Output',exist_ok=True)


spark=SparkSession.builder.appName('Recession Predictor').master('local').getOrCreate()

feature_cols = ['FEDFUNDS', 'DGS10', 'UNRATE', 'MORTGAGE30US', 'is_recession','M2SL','CPI']
ml_feature_cols=['FEDFUNDS', 'DGS10', 'UNRATE', 'MORTGAGE30US', 'yield_spread','M2SL','CPI','FedRates_3_months','Unrates_3_months','yield_spread_3_months']

def feature_engineering():
    silver_df=spark.read.format('csv').option('header',True).option('inferSchema',True).load('Datasets/recession_predictor.csv')
    filtered_df=silver_df.dropna(subset=feature_cols).withColumn('yield_spread',(F.col('DGS10')-F.col('FEDFUNDS')))
    w=Window().orderBy(F.col('obs_date'))
    filtered_df=filtered_df.withColumn('FedRates_3_months',F.lag('FEDFUNDS',3).over(w)).withColumn('Unrates_3_months',F.lag('UNRATE',3).over(w)).withColumn('yield_spread_3_months',F.lag('yield_spread',3).over(w)).dropna(subset=['FedRates_3_months','Unrates_3_months','yield_spread_3_months'])
    return filtered_df
filtered_df=feature_engineering()

def model_preprocessor(filtered_df):
    assembler=VectorAssembler(inputCols=ml_feature_cols,outputCol='features')
    assembled_df = assembler.transform(filtered_df)
    return assembled_df
assembled_df=model_preprocessor(filtered_df)

with mlflow.start_run(run_name="recession_predictor_v1"):
    train_df, test_df = assembled_df.randomSplit(weights=[0.8, 0.2], seed=42)
    train_df=train_df.withColumn('weight',F.when(F.col('is_recession')==1,7.6).otherwise(1))
    test_df=test_df.withColumn('weight',F.when(F.col('is_recession')==1,7.6).otherwise(1))
    model=LogisticRegression(featuresCol='features',labelCol='is_recession',weightCol='weight')
    lr_model=model.fit(train_df)
    predictions=lr_model.transform(test_df)
    bc_evaluator_auc=BinaryClassificationEvaluator(labelCol='is_recession',metricName='areaUnderROC')
    mc_evaluator_accuracy=MulticlassClassificationEvaluator(labelCol='is_recession',metricName='accuracy')
    mc_evaluator_recall = MulticlassClassificationEvaluator(labelCol='is_recession', metricName='weightedRecall')
    mc_evaluator_precision = MulticlassClassificationEvaluator(labelCol='is_recession', metricName='weightedPrecision')
    mlflow.log_param("class_weight", 7.6)
    mlflow.log_param("train_split", 0.8)
    mlflow.log_param("test_split", 0.2)
    mlflow.log_param("seed", 42)
    mlflow.log_param("features", str(ml_feature_cols))
    mlflow.log_metric('auc',bc_evaluator_auc.evaluate(predictions))
    mlflow.log_metric('accuracy',mc_evaluator_accuracy.evaluate(predictions))
    mlflow.log_metric('recall',mc_evaluator_recall.evaluate(predictions))
    mlflow.log_metric('precision',mc_evaluator_precision.evaluate(predictions))
    signature = mlflow.models.infer_signature(assembled_df.select(ml_feature_cols).limit(5).toPandas(),predictions.select('prediction').limit(5).toPandas())
    mlflow.spark.log_model(lr_model,"recession_model",registered_model_name="US_Recession_Predictor",signature=signature,pip_requirements=["pyspark==3.5.0"])
run_id = mlflow.last_active_run().info.run_id
model_uri = f"runs:/{run_id}/recession_model"  

def score_new_data(model_uri):
    loaded_model=mlflow.spark.load_model(model_uri)
    filtered_df=feature_engineering()
    assembled_df=model_preprocessor(filtered_df)
    predictions_df=loaded_model.transform(assembled_df)
    final_df=predictions_df.select(
        F.col('obs_date').alias('obs_date'),
        F.col('is_recession').alias('is_recession'),
        F.col('prediction').cast('int').alias('is_recession_predicted'),
        vector_to_array(F.col("probability")).getItem(1).cast('double').alias('recession_probability'),
        F.current_timestamp().alias('predicted_at')
    )
    final_df.write.option('header',True).mode('overwrite').csv('Output/model_score/')
          
score_new_data(model_uri)

print('Trained, Scored and Saved the model. Opening UI on Browser....')
subprocess.Popen(["mlflow", "ui", "--port", "5001"])
time.sleep(2)
webbrowser.open("http://127.0.0.1:5001")