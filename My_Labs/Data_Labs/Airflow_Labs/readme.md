# About
In this Lab, I try to read a Housing Price csv, perform data preprocessing on it, train a Linear Regression Model and save the model
along with the performance metrics such as R2 score, RMSE, Absolue Mean etc, and the Predicted values.

## Working
First, I changed the docker image to add the additional folders needed. Next, I created the helper file which has 4 functions. The first
function just reads the file and returns the output as a dataframe. The second function reads the dataframe and performs data preprocessing
such as dropping duplicates, null values and encodes the string columns, after which it returns the dataframe. The third function takes the
dataframe and uses it to train the regression model, gets the metrics and the y_pred and returns it all as a metrics dictionary. The fourth 
function takes this metrics dictionary and writes it into a txt file.

The airflow dag is defined in four steps which have python operators to execute each of the four functions above. Each output of the function 
is saved as XCOMS which are accessed via the task_id.output to use as arguments for the next task.


## How to run the code
The helper code can be run in the local system as long as the model,src and data folders are present and the code itself is saved within the
src folder. The airflow file needs a few changes in the docker-compose.yaml file.
- ${AIRFLOW_PROJ_DIR:-.}/src:/opt/airflow/src
- ${AIRFLOW_PROJ_DIR:-.}/data:/opt/airflow/data
- ${AIRFLOW_PROJ_DIR:-.}/model:/opt/airflow/model

All these need to be added to the yaml under volumes and  docker needs to be restarted. After this the airflow file should be moved under dags folder and the dag will appear in the airflow ui.


## Execution
Final Run <img width="1470" height="956" alt="Screenshot 2026-02-13 at 9 08 17 PM" src="https://github.com/user-attachments/assets/3718c892-9774-4cbe-a14c-642b54f7f3ba" />
