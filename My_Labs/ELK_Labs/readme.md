# ELK LAB
## About
In this lab, I built an end to end fraud detection pipeline on a 50k financial transaction dataset. The pipeline preprocesses raw transaction data, trains 10 ML models using RandomizedSearchCV, selects  best model, and exports predictions to Elasticsearch using Logstash. Model evaluation metrics are saved as logs using Filebeat and Docker container/system health is monitored via Metricbeat. All three data streams are visualized using Kibana dashboards.

## Working
### ML Pipeline:
The dataset is downloaded from Kaggle using kagglehub and preprocessed by dropping identifier columns, extracting datetime features, encoding categoricals with LabelEncoder, mapping binary flags, and scaling with StandardScaler. The target variable Fraud_Label has a ~4.8% fraud rate, so scale_pos_weight is tuned for XGBoost and class_weight="balanced" is used across other models. Ten models are trained using RandomizedSearchCV with StratifiedKFold(n_splits=3) and scored on roc_auc. The best model is selected dynamically, predictions and fraud probabilities are exported as NDJSON, and evaluation metrics (F1, confusion matrix, FP/FN rates, feature importances) are written to a log file.
### ELK Ingestion:
- Logstash reads the NDJSON predictions file and indexes each transaction record into an Elasticsearch index called fraud predictions
- Filebeat uses a filestream input to monitor the model log file and ships each log line into a Filebeat data stream
- Metricbeat collects system CPU, memory, disk I/O, and network metrics alongside per container Docker metrics every 10 seconds
- All five services: Elasticsearch, Kibana, Logstash, Filebeat, and Metricbeat, are orchestrated via Docker Compose.

## HOW TO REPLICATE
- Clone the github repo
- Navigate to  ELK_Labs folder
- Run the following command: python ELK_Lab.py and wait for the run to be completed
- Start docker service: docker compose up -d
- Verify data ingestion by running:

curl -s "http://localhost:9200/fraud-predictions/_count"

curl -s "http://localhost:9200/.ds-filebeat-*/_count"

curl -s "http://localhost:9200/metricbeat-*/_count"

## Requirements
Docker + Docker Compose
Python 3.10+
Kaggle API credentials (for dataset download)
