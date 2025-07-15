import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px # reminder this save html
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
import utils
import os

url = "https://cf-courses-data.s3.us.cloud-object-storage.appdomain.cloud/IBM-ML0232EN-SkillsNetwork/asset/airlines_data.xlsx"

os.makedirs("data", exist_ok=True)
path = "data/airline_data.xlsx"
print(utils.checkIfAssetExists(path))
if not utils.checkIfAssetExists(path):
    utils.download(url, path)

data = pd.read_excel(path)
print(data.head())

print(data.info())
# Provides stat information about numerical values (only have 1: "Price")
print(data.describe())

# Null value check
print(data.isnull().sum())

# Fill any null value with last observed non-null value until another non-null encountered
data.ffill(inplace=True)
print('=========')
print(data.isnull().sum())

# Feature Transformation
# Dealing with Categorical Variables

# use unique() to obtain all categories in "Airlines"
airlines = data["Airline"].unique().tolist()
print(airlines)
# Some airline names repeated ex "Jet Airways" "Jet Airways Business"; some of airlines are subdivided into separate parts
# Combine two part airlines to make categorical features more consistent with the rest of the values\
# numpy where() to locate and combine the two categories
data["Airline"] = np.where(data["Airline"] == 'Vistara Premium economy', 'Vistara', data["Airline"])
data["Airline"] = np.where(data["Airline"] == 'Jet Airways Business', 'Jet Airways', data["Airline"])
data["Airline"] = np.where(data["Airline"] == 'Multiple carriers Premium economy', 'Multiple carriers', data["Airline"])

airlines = data["Airline"].unique().tolist()
print(airlines)

# One-Hot Encoding - convert categorical to numerical
# get_dummies() for transformation. Transform 'Airline', 'Source', 'Destination' to numerical values
data1 = pd.get_dummies(data=data, columns = ["Airline", "Source", "Destination"])
print(data1.head())
print(f"og_data: {data.shape}") # 11 original features
print(f"hot_data: {data1.shape}") # 38 new fts after hot encoding 
print(data1["Total_Stops"].value_counts())

# Label Encoding - values manually assigned to corresponding keys
# "Total_Stops": non-stop - 0; 1 stop: 1 etc
stop_mapping = {
    'non-stop': 0,
    '1 stop': 1,
    '2 stops': 2,
    '3 stops': 3,
    '4 stops': 4,
}
# pd.set_option('future.no_silent_downcasting', True)
data1.replace(stop_mapping, inplace=True) # .infer_objects(copy=False) will give back the original int64 rather than being typecast as an object
print(data1["Total_Stops"].head())

# Date Time Transformations
# Transform the 'Duration' time column; string -> numerical
# iterate through 'Duration' split to hrs and min as 2 additional separate cols.
# Add 'Duration_hours' in min; 'Duration_min' col to get a 'Duration_total_mins'. Useful for regression type of analysis
duration = list(data1['Duration'])
for i in range(len(duration)):
    if len(duration[i].split()) != 2:
        if 'h' in duration[i]:
            duration[i] = duration[i].strip() + ' 0m'
        elif 'm' in duration[i]:
            duration[i] = f"0h {duration[i].strip()}"
dur_hours = []
dur_minutes = []
for i in range(len(duration)):
    dur_hours.append(int(duration[i].split()[0][:-1]))
    dur_minutes.append(int(duration[i].split()[1][:-1]))

data1['Duration_hours'] = dur_hours
data1['Duration_minutes'] = dur_minutes
data1.loc[:, 'Duration_hours'] *= 60
data1['Duration_Total_mins'] = data1['Duration_hours'] + data1['Duration_minutes']
print(data1.loc[:,['Duration_hours','Duration_minutes','Duration_Total_mins']].head())

# Transform 'Departure', 'Arrival'
data1["Dep_Hour"] = pd.to_datetime(data1["Dep_Time"]).dt.hour
data1["Dep_Min"] = pd.to_datetime(data1["Dep_Time"]).dt.minute
data1["Arrival_Hour"] = pd.to_datetime(data1["Arrival_Time"]).dt.hour
data1["Arrival_Min"] = pd.to_datetime(data1["Arrival_Time"]).dt.minute

# Split Departure and Arrival times into time zones
data1['dep_timezone'] = pd.cut(data1["Dep_Hour"], [0,6,12,18,24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
print(data1["dep_timezone"])
data1['arrival_timezone'] = pd.cut(data1["Arrival_Hour"], [0,6,12,18,24], labels=['Night', 'Morning', 'Afternoon', 'Evening'])
print(f"arrivals\n {data1["arrival_timezone"]}")

timezone_mapping = {'Night': 0, 'Morning': 1, 'Afternoon': 2, 'Evening': 3}
data1['dep_timezone_encoded'] = data1["dep_timezone"].map(timezone_mapping)
data1['arrival_timezone_encoded'] = data1["arrival_timezone"].map(timezone_mapping)

# Transform 'Date_of_Journey'. Og dtype is an object that MLs can't use yet
# create a new month column, extract the month from  the date_of_journey
data1['Month'] = pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.month
# extract day from date_of_journey
data1['Day'] = pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.day
# extract year from date_of_journey
data1['Year'] = pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.year
# extract the day of the week from date_of_journey
data1["day_of_week"] = pd.to_datetime(data1["Date_of_Journey"], format="%d/%m/%Y").dt.day_name()

# Feature Selection: choose attributes which best explain the relationship of the independent variables with respect to the target variable: "Price"
# Building a heatmap and calculating the correlation coefficients scores are most common

# Select only relevant and newly transformed variables; exclude 'Route', 'Additional_info', and all original categorical variables.
# Place into a 'new_data' DataFrame
print(data1.columns)

heatmap_feature_cols = ['Total_Stops', 'Airline_Air Asia',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Multiple carriers', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Source_Banglore',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
       'Duration_hours', 'Duration_minutes', 'Duration_Total_mins', 'Dep_Hour',
       'Dep_Min', 'dep_timezone_encoded', 'arrival_timezone_encoded', 'Price']
heatmap_data = data1.loc[:, heatmap_feature_cols]

os.makedirs('plots', exist_ok=True)

plt.figure(figsize=(18,18))
# .corr() - calculates the pairwise correlation b/w numerical cols in a DataFrame. Outputs a val representing how strong 2 vars are linearly related
# Pearson correlation coefficient (linear relationship). 1: pos relation; -1: neg relation; 0: no linear correlation
sns.heatmap(heatmap_data.corr(), annot=True, cmap='RdYlGn')
path = "plots/heatmap_features.png"
if not utils.checkIfAssetExists(path):
    utils.savePlot(path)

