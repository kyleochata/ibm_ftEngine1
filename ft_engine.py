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