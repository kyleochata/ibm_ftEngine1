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

# use corr() to calculate and list correlation b/w all independent variables and 'price'
features = heatmap_data.corr()["Price"].sort_values(ascending=False)
print(f"Features_v_Price\n {features}")
# pos relationship moves the indpendent var and Price in the same direction
# neg: increase in 1 decreases the other
plt.figure(figsize=(20, 10))
plt.bar(features.index, features.to_numpy())
plt.xticks(rotation=90, ha='center')
plt.title("Feature Correlations with Price", fontsize=14)
plt.ylabel("Correlation value to Price", fontsize=12)
plt.xlabel("Features")
plt.grid(axis='y', linestyle="--", alpha=0.7)
plt.tight_layout()
path = "plots/price_correlation.png"
if not utils.checkIfAssetExists(path):
    utils.savePlot(path)

# Feature Extraction using Principal Component Analysis
# Dimentionality reduction - part of ft extraction process that combines the existing fts to produce more useful ones
# Goal: simplify the data w/o loosing too much info
# Principal Component Analysis (PCA) - id hyperplane that lies closest to the data, then projects data onto it. Few multidimensional fts merged into one

# scale data. Assign all independent vars to x and dependent var ('price') to y
x = data1.loc[:,['Total_Stops', 'Airline_Air Asia',
       'Airline_Air India', 'Airline_GoAir', 'Airline_IndiGo',
       'Airline_Jet Airways', 'Airline_Multiple carriers', 'Airline_SpiceJet',
       'Airline_Trujet', 'Airline_Vistara', 'Source_Banglore',
       'Source_Chennai', 'Source_Delhi', 'Source_Kolkata', 'Source_Mumbai',
       'Destination_Banglore', 'Destination_Cochin', 'Destination_Delhi',
       'Destination_Hyderabad', 'Destination_Kolkata', 'Destination_New Delhi',
       'Duration_hours', 'Duration_minutes', 'Duration_Total_mins', 'Dep_Hour',
       'Dep_Min']]
ft_names = x.copy()
y = data1["Price"] # target want to predict
# Many fts in x are on different scales. PCA assumes all fts are centered and scaled
# ex: 'Duration_Total_mins' scale is 0-1000+ and will dominate other values like our hot-encoded binary or label encoded (dep/arrival_timezones)
# StandardScaler(): transforms ea ft to have: mean = 0 (centered around 0); std dev =1 (unit variance)
# scaled_val = (x-mean) / std_dev
# By standardizing the independent features variables ensures that each one contributes fairly to the model's predictions to prevent larger raw values from dominating those with smaller vals
# By standardizing, you’re essentially asking:
# "Holding all else equal, which features—when they vary by one standard deviation—actually move the needle on price?"
scaler = StandardScaler()
x = scaler.fit_transform(x.astype(np.float64)) #.astype(np.float64) ensures all values are numeric
print(x)

# Apply fit_transform on x again to reduce dimensionality to 2 dimensions
pca = PCA(n_components=2)
pca.fit_transform(x)
print(f"pca:\n {pca}")

# Explained Variance Ratio: indicates the proportion of the dataset's variance that lies along each principal component
explained_variance = pca.explained_variance_ratio_
print(sum(explained_variance))
#explained_variance[0]: 17.54% of the variance; [1]: 12.11% of variance = PCA only captures ~30% of original data's patterns
# Means that we lost too much data (variance) by trying to compress down to 2D

pca = PCA(n_components=7)
pca.fit_transform(x)
explained_variance = pca.explained_variance_ratio_
print(sum(explained_variance))

# with plt.style.context('dark_background'):
plt.figure(figsize=(6,4))
plt.bar(range(7), explained_variance, alpha=0.5, align='center', label='individual explained variance')
plt.ylabel('Explained variance ratio')
plt.xlabel('Prinicpal components')
plt.legend(loc='best')
plt.tight_layout()
path = "plots/explainedVariance_v_PrincipalComponents.png"
if not utils.checkIfAssetExists(path):
    utils.savePlot(path)

# Determining the Right Number of Dimensions
# Perform PCA without reducing dimensionality, then compute the min number of diimensions required to preserve 95% variance
pca = PCA()
pca.fit(x)
cumsum = np.cumsum(pca.explained_variance_ratio_)
d = np.argmax(cumsum >= 0.95) + 1
print(d) # 16 n_components needed

# Set n_components to a float b/w 0.0 - 1.0 for what variance wanted to preserve
pca = PCA(n_components=0.95)
x_reduced = pca.fit_transform(x)

fig = px.area(
    x=range(1, cumsum.shape[0] + 1),
    y=cumsum,
    labels={"x": "# Components", "y": "Explained Variance"}
)

path = "plots/pca_variance.html"
if not utils.checkIfAssetExists(path):
    fig.write_html(path)
fig.show()

loadings = pd.DataFrame(
    pca.components_.T,  # Transpose to align features as rows
    columns=[f'PC{i}' for i in range(1, pca.n_components_ + 1)],
    index=ft_names.columns    # Original feature names
)

for pc in loadings.columns:
    print(f"\n Top fts for {pc}")
    print(loadings[pc].abs().sort_values(ascending=False).head())