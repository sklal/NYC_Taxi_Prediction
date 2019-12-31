#EDA
#Importing Libraries
#%%
import numpy as np
import pandas as pd
from datetime import timedelta
import datetime as dt
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
import warnings
warnings.filterwarnings('ignore')

#Data Loading
#%%
df = pd.read_csv('nyc_taxi_final')