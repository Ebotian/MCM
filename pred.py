import pandas as pd

############ load and digitize the data
data=pd.read_csv('/home/ebotian/MCM/tennis.csv')
data = pd.get_dummies(data, columns=['winner_shot_type','serve_width','serve_depth','return_depth'])
############

############# pre-process the data
# fill nan with 0, and replace AD with 50
data = data.fillna(0)
data = data.replace('AD', 50.0)
data['point_victor']=data["point_victor"].replace(2,0)
#print(data.iloc[:,15])

# split the data into different match
grouped = dict(tuple(data.groupby(data['match_id'].ne(data['match_id'].shift()).cumsum())))

# Rename the subdata
subdata = {df['match_id'].iloc[0]: df for _, df in grouped.items()}

# Create a new dataset from the first column, excluding duplicates
match = pd.DataFrame(data.iloc[:, 0].drop_duplicates()).iloc[:,0].tolist()
#print(match_id[0])
##############
id=0
##############
# Calculate the time difference between consecutive rows

# Convert the timestamp column to datetime format
subdata[match[id]]['elapsed_time'] = pd.to_timedelta(subdata[match[id]]['elapsed_time'])

# Calculate the time difference between consecutive rows
subdata[match[id]]['time_diff'] = subdata[match[id]]['elapsed_time'].diff()

# Calculate the 5th and 95th percentiles
lower_threshold = subdata[match[id]]['time_diff'].quantile(0.05)
upper_threshold = subdata[match[id]]['time_diff'].quantile(0.95)

# Exclude the top 5% and bottom 5% of periods
filtered_diff = subdata[match[id]]['time_diff'][(subdata[match[id]]['time_diff'] > lower_threshold) & (subdata[match[id]]['time_diff'] < upper_threshold)]

# Calculate the average of the remaining intervals
average_interval = filtered_diff.mean()

print(f'Average interval (excluding top 5% and bottom 5%): {average_interval}')

# Convert the time differences to integer seconds
subdata[match[id]]['time_diff'] = subdata[match[id]]['time_diff'].dt.total_seconds()

# Replace 'NaT' values with 0
subdata[match[id]]['time_diff'] = subdata[match[id]]['time_diff'].fillna(0).astype(int)
#fill nan with 0
#subdata[match[id]]['time_diff'] = subdata[match[id]]['time_diff'].fillna(0)
############ add features
add_feature=["score_diff"]

########## defining the new features

#subdata[match_id[0]][add_feature[0]] = subdata[match_id[0]]['p1_games'] - subdata[match_id[0]]['p2_games']
# split the data into features and target
target=pd.DataFrame(subdata[match[id]]["point_victor"])
# Add the "elapsed_time" column to the "target" DataFrame
target.insert(0, 'elapsed_time', subdata[match[id]]['elapsed_time'])
target['elapsed_time'] = target['elapsed_time'].dt.total_seconds()

subdata[match[id]]=subdata[match[id]].drop(columns=["point_victor"])
features=subdata[match[id]].iloc[:,4:]
#print(target)
from re import T
from statsmodels.tsa.arima.model import ARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_squared_error
from math import sqrt
from sklearn.linear_model import LinearRegression

# Split data into training set and test set
train_size = int(len(target) * 0.7)
train, test = target[0:train_size], target[train_size:len(target)]
train=train.astype(float)
print(train)
print(test)
import numpy as np
from pydtmc import MarkovChain

# Prepare the data
train_data = train["point_victor"].values

# Calculate the transition matrix
transition_matrix = np.zeros((2, 2))
for i in range(len(train_data) - 1):
    transition_matrix[int(train_data[i]), int(train_data[i+1])] += 1
transition_matrix /= transition_matrix.sum(axis=1, keepdims=True)

# Create the Markov Chain
mc = MarkovChain(transition_matrix, ['0', '1'])

# Function to predict the next state
def predict_next_state(mc, current_state):
    return np.random.choice(mc.states, p=mc.p[int(current_state)])

# Predict the next state for the test data
test_data = test["point_victor"].values

# Get the last state from the training data
last_state_train = train_data[-1]

# Generate the first prediction from the last state of the training data
first_prediction = predict_next_state(mc, last_state_train)

# Generate the rest of the predictions from the test data
predictions_markov = [predict_next_state(mc, state) for state in test_data[:-1]]

# Insert the first prediction at the beginning of the predictions list
predictions_markov.insert(0, first_prediction)

#print(predictions)
#print(test_data[1:])

# Calculate the RMSE
rmse_markov = mean_squared_error(test_data[:], np.array(predictions_markov).astype(float), squared=False)

print(rmse_markov)

# Calculate the number of same values
same_values = (test_data[:] == np.array(predictions_markov).astype(float)).sum()

# Calculate the percentage
percentage_markov = same_values / len(test_data[:]) * 100

print(f"Number of same values: {same_values}")
print(f"Percentage: {percentage_markov}%")

# ARIMA
model_arima = ARIMA(train["point_victor"], order=(5,1,0))
model_arima_fit = model_arima.fit()
predictions_arima = model_arima_fit.forecast(steps=len(test))


#print(forecast_output)
#print(test["point_victor"])
rmse_arima = sqrt(mean_squared_error(test["point_victor"], predictions_arima))
#print(rmse_arima)

# Convert the ARIMA predictions
forecast_output_binary = np.where(predictions_arima > 0.5, 1, 0)

# Calculate the RMSE
rmse_arima_binary = sqrt(mean_squared_error(test["point_victor"], forecast_output_binary))

# Calculate the number of same values
same_values_arima = (test["point_victor"] == forecast_output_binary).sum()

# Calculate the percentage
percentage_arima = same_values_arima / len(test["point_victor"]) * 100

print(f"RMSE: {rmse_arima_binary}")
print(f"Number of same values: {same_values_arima}")
print(f"Percentage: {percentage_arima}%")

# SARIMA
model_sarima = SARIMAX(train["point_victor"], order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
model_sarima_fit = model_sarima.fit(disp=0)
predictions_sarima = model_sarima_fit.forecast(steps=len(test))

rmse_sarima = sqrt(mean_squared_error(test["point_victor"], predictions_sarima))
print(rmse_sarima)

# Convert the SARIMA predictions
predictions_sarima_binary = np.where(predictions_sarima > 0.5, 1, 0)

# Calculate the RMSE
rmse_sarima_binary = sqrt(mean_squared_error(test["point_victor"], predictions_sarima_binary))

# Calculate the number of same values
same_values_sarima = (test["point_victor"] == predictions_sarima_binary).sum()

# Calculate the percentage
percentage_sarima = same_values_sarima / len(test["point_victor"]) * 100

print(f"RMSE: {rmse_sarima_binary}")
print(f"Number of same values: {same_values_sarima}")
print(f"Percentage: {percentage_sarima}%")
# Calculate the weights
total_weight=(abs(percentage_markov / 100 - 0.5) + abs(percentage_arima / 100 - 0.5) + abs(percentage_sarima / 100 - 0.5))
weight_markov = abs(percentage_markov / 100 - 0.5) / total_weight
weight_arima = abs(percentage_arima / 100 - 0.5) / total_weight
weight_sarima = abs(percentage_sarima / 100 - 0.5)/ total_weight
# Reverse the predictions if the accuracy is less than 50%
if percentage_markov < 50:
    predictions_markov = [1.0 - float(p) for p in predictions_markov]
if percentage_arima < 50:
    predictions_arima = [1.0 - p for p in predictions_arima]
if percentage_sarima < 50:
    predictions_sarima = [1.0 - p for p in predictions_sarima]

print(predictions_markov)
# Convert lists to numpy arrays
predictions_markov = np.array(predictions_markov).astype(float)
predictions_arima = np.array(predictions_arima)
predictions_sarima = np.array(predictions_sarima)

# Calculate the combined predictions
combined_predictions = weight_markov * predictions_markov + weight_arima * predictions_arima + weight_sarima * predictions_sarima

# Convert the combined predictions to binary
combined_predictions_binary = np.where(combined_predictions > 0.5, 1, 0)

# Calculate the RMSE
rmse_combined = sqrt(mean_squared_error(test["point_victor"], combined_predictions_binary))

# Calculate the number of same values
same_values_combined = (test["point_victor"] == combined_predictions_binary).sum()

# Calculate the percentage
percentage_combined = same_values_combined / len(test["point_victor"]) * 100

print(f"RMSE: {rmse_combined}")
print(f"Number of same values: {same_values_combined}")
print(f"Percentage: {percentage_combined}%")

#list all three alone percentage
print(f"Percentage of Markov: {percentage_markov}%")
print(f"Percentage of ARIMA: {percentage_arima}%")
print(f"Percentage of SARIMA: {percentage_sarima}%")
# Print RMSE values
print('RMSE values:')
print('Markov Chain: ', rmse_markov)
print('ARIMA: ', rmse_arima)
print('SARIMA: ', rmse_sarima)
#print('LSTM: ', rmse_lstm)
