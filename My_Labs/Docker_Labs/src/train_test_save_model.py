from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.preprocessing import LabelEncoder,StandardScaler
from sklearn.model_selection import RandomizedSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.pipeline import Pipeline
from sklearn.metrics import confusion_matrix, precision_score, recall_score, f1_score
from xgboost import XGBClassifier
import pandas as pd
import numpy as np
import joblib
import matplotlib
matplotlib.use('Agg') 
import matplotlib.pyplot as plt
import seaborn as sns
import os


os.makedirs('output/Model_Selection', exist_ok=True)
os.makedirs('output/Evaluation_Metrics', exist_ok=True)
os.makedirs('output/model', exist_ok=True)
os.makedirs('output/Confusion_Matrix', exist_ok=True)


df=pd.read_csv('train.csv')

cols_to_encode=['gender','ethnicity','education_level','income_level','smoking_status','employment_status']
encoders = {}
le = LabelEncoder()
for col in cols_to_encode:
    df[col] = le.fit_transform(df[col])
    encoders[col] = le
    le = LabelEncoder() 

joblib.dump(encoders, 'output/model/Label_Encoders') 

log_reg_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', LogisticRegression(solver='saga', max_iter=1000, random_state=42, penalty='elasticnet'))
])
knn_pipeline = Pipeline([
    ('scaler', StandardScaler()),
    ('model', KNeighborsClassifier(n_jobs=1))
])

params_grid={
    'LogisticRegression':{
        'model':log_reg_pipeline,
        'params':{
        'model__C': [0.01, 0.1, 1, 10],
        'model__class_weight': [None, 'balanced'],
        'model__l1_ratio': [0, 1]
        }
    },
    'KNeighborsClassifier':{
        'model':knn_pipeline,
        'params':{
            'model__n_neighbors': [5, 15, 31],
            'model__weights': ['uniform'],
            'model__metric': ['euclidean']
        }
    },
    'DecisionTreeClassifier':{
        'model':DecisionTreeClassifier(),
        'params':{
            'max_depth':[10, 15, 20, None],
            'min_samples_leaf': [1, 5, 10],
            'criterion':['gini','entropy']
        }
    },
    'RandomForestClassifier':{
        'model':RandomForestClassifier(n_jobs=1),
        'params':{
            'n_estimators': [200, 500],
            'max_depth': [10, 15, None],
            'min_samples_leaf': [1, 5, 10],
            'max_features': ['sqrt']
        }
    },
    'GaussianNB':{
        'model':GaussianNB(),
        'params':{
            'var_smoothing':[1e-10, 1e-9, 1e-8]
        }
    },
    'xgboost':{
        'model':XGBClassifier(tree_method='hist',n_jobs=1,eval_metric='logloss',random_state=42, verbosity=0),
        'params':{
            'n_estimators': [200],
            'max_depth': [3, 5],
            'learning_rate': [0.05, 0.1]
            }
    }
}

X=df.drop(columns=['diagnosed_diabetes'])
y=df.diagnosed_diabetes

sample_idx = X.sample(frac=0.5, random_state=42).index
X_sample = X.loc[sample_idx]
y_sample = y.loc[sample_idx]

X_train,X_test,y_train,y_test=train_test_split(X,y,test_size=0.2,random_state=42)

scores=[]
for model,params in params_grid.items():
    print(f"Starting model search for {model}...")
    clf=RandomizedSearchCV(params['model'],params['params'],n_iter=3,verbose=1,n_jobs=1,random_state=42, cv=3, return_train_score=False)
    clf.fit(X_sample,y_sample)
    scores.append({'model':clf.best_estimator_,'params':clf.best_params_,'score':clf.best_score_})

clf_df=pd.DataFrame(scores,columns=['model','params','score'])

clf_df.to_csv('output/Model_Selection/Best_Model.csv')

model=clf_df.sort_values(by='score',ascending=False).reset_index()['model'][0]

print(f"Training {model}...")

model.fit(X_train,y_train)

y_pred=model.predict(X_test)

cm=confusion_matrix(y_true=y_test,y_pred=y_pred)

p_score=precision_score(y_test,y_pred)

r_score=recall_score(y_test,y_pred)

f1score=f1_score(y_test,y_pred)

pd.DataFrame([{
    'Precision Score':p_score,
    'recall_score':r_score,
    'f1_score':f1score
}]).to_csv('output/Evaluation_Metrics/Evaluation_Metrics.csv')

joblib.dump(model,'output/model/Diabetes_Model')

plt.figure(figsize=(6,6))
plt.title('Confusion Matrix')
sns.heatmap(cm,annot=True)
plt.savefig('output/Confusion_Matrix/confusion_matrix.png',dpi=300, bbox_inches='tight')