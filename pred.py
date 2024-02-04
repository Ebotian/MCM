import pandas as pd
import numpy as np

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
##############
#invert the victor when the server is 2 to get server_victor
#and after the prediction, we invert the victor again
# Get the index array
index_array = subdata[match[id]][subdata[match[id]]['server'] == 2].index.values

# Invert the values in the "point_victor" column for the specified rows
subdata[match[id]].loc[index_array, 'point_victor'] = 1 - subdata[match[id]].loc[index_array, 'point_victor']
#print(index_array)
##############
def get_average_interval(id, subdata):
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

    return average_interval
## Convert the timestamp column to datetime format
#subdata[match[id]]['elapsed_time'] = pd.to_timedelta(subdata[match[id]]['elapsed_time'])
#
## Calculate the time difference between consecutive rows
#subdata[match[id]]['time_diff'] = subdata[match[id]]['elapsed_time'].diff()
#
## Calculate the 5th and 95th percentiles
#lower_threshold = subdata[match[id]]['time_diff'].quantile(0.05)
#upper_threshold = subdata[match[id]]['time_diff'].quantile(0.95)
#
## Exclude the top 5% and bottom 5% of periods
#filtered_diff = subdata[match[id]]['time_diff'][(subdata[match[id]]['time_diff'] > lower_threshold) & (subdata[match[id]]['time_diff'] < upper_threshold)]
#
## Calculate the average of the remaining intervals
#average_interval = filtered_diff.mean()
#
#print(f'Average interval (excluding top 5% and bottom 5%): {average_interval}')
#
## Convert the time differences to integer seconds
#subdata[match[id]]['time_diff'] = subdata[match[id]]['time_diff'].dt.total_seconds()
#
## Replace 'NaT' values with 0
#subdata[match[id]]['time_diff'] = subdata[match[id]]['time_diff'].fillna(0).astype(int)
##fill nan with 0
############ add features
#add_feature=["score_diff"]

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
#print(train)
#print(test)

#find index
# Get the indices in train that are in index_array
train_indices = np.intersect1d(train.index.values, index_array)

# Get the indices in test that are in index_array
test_indices = np.intersect1d(test.index.values, index_array)

train_array_indices = train_indices - min(train.index.values)
test_array_indices = test_indices - min(test.index.values)


#Markov Chain
from pydtmc import MarkovChain

def train_and_predict_markov(train_data, test_data, test_array_indices):
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

    # Get the last state from the training data
    last_state_train = train_data[-1]

    # Generate the first prediction from the last state of the training data
    first_prediction = predict_next_state(mc, last_state_train)

    # Generate the rest of the predictions from the test data
    predictions_markov = [predict_next_state(mc, state) for state in test_data[:-1]]

    # Insert the first prediction at the beginning of the predictions list
    predictions_markov.insert(0, first_prediction)
    predictions_markov = np.array(predictions_markov).astype(float)

    # Invert the values in predictions and test_data for the specified indices
    predictions_markov[test_array_indices] = 1 - predictions_markov[test_array_indices]
    test_data[test_array_indices] = 1 - test_data[test_array_indices]

    # Calculate the percentage of correct predictions
    same_values_markov = (test_data == predictions_markov).sum()
    percentage_markov = same_values_markov / len(test_data) * 100

    return predictions_markov, percentage_markov

train_data = train["point_victor"].values
test_data = test["point_victor"].values

predictions_markov, percentage_markov = train_and_predict_markov(train_data, test_data, test_array_indices)

# ARIMA
from statsmodels.tsa.arima.model import ARIMA
from sklearn.metrics import mean_squared_error
from math import sqrt

def train_and_predict_arima(train_data, test_data, test_array_indices):
    # Train the ARIMA model
    model_arima = ARIMA(train_data, order=(5,1,0))
    model_arima_fit = model_arima.fit()

    # Make predictions
    predictions_arima = model_arima_fit.forecast(steps=len(test_data))

    # Invert the predictions for the specified indices
    predictions_arima[test_array_indices] = 1 - predictions_arima[test_array_indices]
    test_data[test_array_indices] = 1 - test_data[test_array_indices]

    # Convert the ARIMA predictions to binary
    predictions_arima_binary = np.where(predictions_arima > 0.5, 1, 0)

    # Calculate the percentage of correct predictions
    same_values_arima = (test_data == predictions_arima_binary).sum()
    percentage_arima = same_values_arima / len(test_data) * 100

    return predictions_arima, percentage_arima

train_data = train["point_victor"].values
test_data = test["point_victor"].values

predictions_arima, percentage_arima = train_and_predict_arima(train_data, test_data, test_array_indices)

# SARIMA
from statsmodels.tsa.statespace.sarimax import SARIMAX

def train_and_predict_sarima(train_data, test_data, test_array_indices):
    # Train the SARIMA model
    model_sarima = SARIMAX(train_data, order=(1, 1, 1), seasonal_order=(0, 0, 0, 0))
    model_sarima_fit = model_sarima.fit(disp=0)

    # Make predictions
    predictions_sarima = model_sarima_fit.forecast(steps=len(test_data))
    predictions_sarima = np.array(predictions_sarima)

    # Invert the predictions for the specified indices
    predictions_sarima[test_array_indices] = 1 - predictions_sarima[test_array_indices]
    test_data[test_array_indices] = 1 - test_data[test_array_indices]

    # Convert the SARIMA predictions to binary
    predictions_sarima_binary = np.where(predictions_sarima > 0.5, 1, 0)

    # Calculate the percentage of correct predictions
    same_values_sarima = (test_data == predictions_sarima_binary).sum()
    percentage_sarima = same_values_sarima / len(test_data) * 100

    return predictions_sarima, percentage_sarima

train_data = train["point_victor"].values
test_data = test["point_victor"].values

predictions_sarima, percentage_sarima = train_and_predict_sarima(train_data, test_data, test_array_indices)

# Calculate the weights
total_weight=(abs(percentage_markov / 100 - 0.5) + abs(percentage_arima / 100 - 0.5) + abs(percentage_sarima / 100 - 0.5))
weight_markov = abs(percentage_markov / 100 - 0.5) / total_weight
weight_arima = abs(percentage_arima / 100 - 0.5) / total_weight
weight_sarima = abs(percentage_sarima / 100 - 0.5)/ total_weight
# Reverse the predictions if the accuracy is less than 50%
if percentage_markov < 50:
    predictions_markov = [1.0 - p for p in predictions_markov]
if percentage_arima < 50:
    predictions_arima = [1.0 - p for p in predictions_arima]
if percentage_sarima < 50:
    predictions_sarima = [1.0 - p for p in predictions_sarima]

#print(predictions_markov)
# Convert lists to numpy arrays
# Convert lists to numpy arrays
predictions_markov = np.array(predictions_markov)
predictions_arima = np.array(predictions_arima)
predictions_sarima = np.array(predictions_sarima)

# Calculate the combined predictions
combined_predictions = weight_markov * predictions_markov + weight_arima * predictions_arima + weight_sarima * predictions_sarima

# Convert the combined predictions to binary
combined_predictions_binary = np.where(combined_predictions > 0.5, 1, 0)

# Calculate the RMSE
rmse_combined = sqrt(mean_squared_error(test_data, combined_predictions_binary))

# Calculate the number of same values
same_values_combined = (test_data == combined_predictions_binary).sum()

# Calculate the percentage
percentage_combined = same_values_combined / len(test_data) * 100

print(f"RMSE: {rmse_combined}")
print(f"Number of same values: {same_values_combined}")
print(f"Percentage: {percentage_combined}%")

#list all three alone percentage
print(f"Percentage of Markov: {percentage_markov}%")
print(f"Percentage of ARIMA: {percentage_arima}%")
print(f"Percentage of SARIMA: {percentage_sarima}%")
# Print RMSE values

#weight calculate
def calculate_weights(percentage_markov, percentage_arima, percentage_sarima):
    # Calculate the total weight
    total_weight = (abs(percentage_markov / 100 - 0.5) + abs(percentage_arima / 100 - 0.5) + abs(percentage_sarima / 100 - 0.5))

    # Calculate the individual weights
    weight_markov = abs(percentage_markov / 100 - 0.5) / total_weight
    weight_arima = abs(percentage_arima / 100 - 0.5) / total_weight
    weight_sarima = abs(percentage_sarima / 100 - 0.5) / total_weight

    return weight_markov, weight_arima, weight_sarima

def reverse_predictions(predictions, percentage):
    # Reverse the predictions if the accuracy is less than 50%
    if percentage < 50:
        predictions = [1.0 - p for p in predictions]
    return predictions

def find_best_train_size(start, end, step,id):
    best_percentage = 0
    best_train_size = 0

    for train_size in np.arange(start, end, step):
        # Split the data into training and test sets
        train_data = subdata[match[id]][:int(len(data) * train_size)].values
        test_data = subdata[match[id]][int(len(data) * train_size):]
        train_data = train["point_victor"].values
        test_data = test["point_victor"].values

        # Train the models and make predictions
        # (replace this with your actual model training and prediction code)
        predictions_markov, percentage_markov = train_and_predict_markov(train_data, test_data, test_array_indices)
        predictions_arima, percentage_arima = train_and_predict_arima(train_data, test_data, test_array_indices)
        predictions_sarima, percentage_sarima = train_and_predict_sarima(train_data, test_data, test_array_indices)

        # Calculate the weights
        weight_markov, weight_arima, weight_sarima = calculate_weights(percentage_markov, percentage_arima, percentage_sarima)

        # Reverse the predictions if the accuracy is less than 50%
        predictions_markov = reverse_predictions(predictions_markov, percentage_markov)
        predictions_arima = reverse_predictions(predictions_arima, percentage_arima)
        predictions_sarima = reverse_predictions(predictions_sarima, percentage_sarima)

        # Convert lists to numpy arrays
        predictions_markov = np.array(predictions_markov)
        predictions_arima = np.array(predictions_arima)
        predictions_sarima = np.array(predictions_sarima)

        # Calculate the combined predictions
        combined_predictions = weight_markov * predictions_markov + weight_arima * predictions_arima + weight_sarima * predictions_sarima

        # Convert the combined predictions to binary
        combined_predictions_binary = np.where(combined_predictions > 0.5, 1, 0)

        # Calculate the percentage
        same_values_combined = (test_data == combined_predictions_binary).sum()
        percentage_combined = same_values_combined / len(test_data) * 100

        # Update best_percentage and best_train_size if this is the highest percentage so far
        if percentage_combined > best_percentage:
            best_percentage = percentage_combined
            best_train_size = train_size

    return best_train_size, best_percentage

start = 0.5  # start of the train size range
end = 0.98  # end of the train size range
step = 0.02  # step size for the train size range
id = 0  # replace this with the actual id

#best_train_size, best_percentage = find_best_train_size(start, end, step, id)

#print(f"Best train size: {best_train_size}, Best percentage: {best_percentage}%")

import asyncio

async def find_best_train_size_async(start, end, step, id):
    # Wrap the synchronous function in a executor
    loop = asyncio.get_event_loop()
    return await loop.run_in_executor(None, find_best_train_size, start, end, step, id)

## Initialize variables to store the best train size and percentage for each index
#best_train_sizes = {}
#best_percentages = {}

## Create a list to hold all the tasks
#tasks = []

## Loop over all indices in the match list
#for id in range(len(match)):
#    # Create a task for this index and add it to the list of tasks
#    task = asyncio.ensure_future(find_best_train_size_async(start, end, step, id))
#    tasks.append(task)
#
## Run all the tasks concurrently
#results = asyncio.get_event_loop().run_until_complete(asyncio.gather(*tasks))
#
## Store the best train size and percentage for each index
#for id in range(len(match)):
#    best_train_sizes[id], best_percentages[id] = results[id]
#
## Print the best train size and percentage for each index
#for id in range(len(match)):
#    print(f"Index: {id}, Best train size: {best_train_sizes[id]}, Best percentage: {best_percentages[id]}%")
#

def generate_predictions(model, id, subdata):
    # Calculate the average interval in seconds
    average_interval_seconds = get_average_interval(id, subdata).total_seconds()

    # Calculate the number of seconds in 6 hours
    six_hours_in_seconds = 6 * 60 * 60

    # Calculate the number of steps that correspond to 6 hours
    steps = int(six_hours_in_seconds / average_interval_seconds)

    # Generate predictions for the next 6 hours
    predictions = model.forecast(steps=steps)

    return predictions