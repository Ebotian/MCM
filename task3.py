import pandas as pd
import numpy as np
from re import T
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score
id=0
data=pd.read_csv('/home/ebotian/MCM/tennis2.csv')

def pre_process(data):
    data = pd.get_dummies(data, columns=['winner_shot_type','serve_width','serve_depth','return_depth'])
    data = data.fillna(0)
    data = data.replace('AD', 50.0)
    grouped = dict(tuple(data.groupby(data['match_id'].ne(data['match_id'].shift()).cumsum())))
    subdata = {df['match_id'].iloc[0]: df for _, df in grouped.items()}
    match = pd.DataFrame(data.iloc[:, 0].drop_duplicates()).iloc[:,0].tolist()
    return subdata,match
subdata,match=pre_process(data)

def process_all_ids(subdata):
    features={}
    target={}
    for id in range(len(match)):
        target[match[id]]=pd.DataFrame(subdata[match[id]]["point_victor"])
        subdata[match[id]]['elapsed_time'] = pd.to_timedelta(subdata[match[id]]['elapsed_time'])
        features[match[id]]=subdata[match[id]].drop(columns=["point_victor"]).iloc[:,4:]
        subdata[match[id]]=subdata[match[id]].drop(columns=["point_victor"])
    return target,features,subdata
target,features,subdata=process_all_ids(subdata)

def get_average_interval(id, subdata):
    subdata[match[id]]['time_diff'] = subdata[match[id]]['elapsed_time'].diff()
    lower_threshold = subdata[match[id]]['time_diff'].quantile(0.05)
    upper_threshold = subdata[match[id]]['time_diff'].quantile(0.95)
    filtered_diff = subdata[match[id]]['time_diff'][(subdata[match[id]]['time_diff'] > lower_threshold) & (subdata[match[id]]['time_diff'] < upper_threshold)]
    average_interval = filtered_diff.mean()
    return average_interval.total_seconds()

average_interval_time=[]
for id in range(len(match)):
    average_interval_time.append(get_average_interval(id, subdata))


X_train={}
X_test={}
y_train={}
y_test={}
for id in range(len(match)):
  X_train[match[id]], X_test[match[id]], y_train[match[id]], y_test[match[id]] = train_test_split(features[match[id]], target[match[id]], test_size=0.3, random_state=42)


classifier={}
X_test_scaled={}
for id in range(len(match)):
  scaler = StandardScaler()
  X_train_scaled = scaler.fit_transform(X_train[match[id]].values)
  X_test_scaled[match[id]] = scaler.transform(X_test[match[id]].values)
  classifier[match[id]] = RandomForestClassifier(n_estimators=100, random_state=42)
  classifier[match[id]].fit(X_train_scaled, y_train[match[id]].values.ravel())


y_pred={}
for id in range(len(match)):
  y_pred[match[id]] = classifier[match[id]].predict(X_test_scaled[match[id]])


accuracy={}
importances={}
for id in range(len(match)):
  accuracy[match[id]] = accuracy_score(y_test[match[id]], y_pred[match[id]])
  importances[match[id]] = classifier[match[id]].feature_importances_
  feature_names = X_train[match[id]].columns
  feature_importances = pd.Series(importances, index=feature_names).sort_values(ascending=False)

print(feature_importances)