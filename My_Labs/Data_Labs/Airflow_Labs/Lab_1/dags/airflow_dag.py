import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import r2_score,mean_squared_error,mean_absolute_error
import joblib
import os

def load_csv():
    df=pd.read_csv(os.path.join(os.path.dirname(__file__), "../data/Housing.csv"))
    return df

def data_preprocessing(df):
    df.dropna(inplace=True)
    df.drop_duplicates(inplace=True)
    columns=['mainroad','guestroom','basement','hotwaterheating','airconditioning','prefarea','furnishingstatus']
    for col in columns:
        le=LabelEncoder()
        df[col]=le.fit_transform(df[col])
    return df

def build_model(df,filename):
    y=df['price']
    x=df.drop(['price'],axis=1)
    X_train,X_test,y_train,y_test=train_test_split(x,y,test_size=0.2,random_state=42)
    model=LinearRegression()
    model.fit(X_train,y_train)
    y_pred=model.predict(X_test)
    r2=r2_score(y_test,y_pred)
    mse=mean_squared_error(y_test,y_pred)
    rmse = np.sqrt(mse)
    mae=mean_absolute_error(y_test,y_pred)
    output_dir=os.path.join(os.path.dirname(os.path.dirname(__file__)), "..model")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, filename)
    joblib.dump(model,output_path)
    metrics = {
        "R2": r2,
        "MSE": mse,
        "RMSE": rmse,
        "MAE": mae,
        'Top 10 Predictions':y_pred[:10].tolist()
    }
    return metrics

def save_results(metrics):
    path=os.path.join(os.path.dirname(__file__), "../model", "model_metrics.txt")
    with open(path, "w") as f:
        f.write("Model Metrics\n")
        for k, v in metrics.items():
            f.write(f"{k}: {v}\n")

if __name__ == "__main__":
    df = load_csv()
    df = data_preprocessing(df)
    metrics= build_model(df, "house_price_model.pkl")
    save_results(metrics)
    print("Training complete. Metrics saved.")
