#Importing the necessary pacakages
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from scipy.ndimage import rotate

pd.set_option('display.max_columns', None)
pd.set_option('display.max_rows', None)
pd.set_option('display.width', 1000)
#Importing the data

data = pd.read_csv(r"C:\Users\igwec\OneDrive\Documents\Desktop\Portfolio Work\Yunu bikes\datasets\YUNU DUPLICATE.csv")

#Checking the data to make sure it accurately imported
print(data.head())

#Checking dataset information
print(data.info())
print()
#Check for missing rows
print(data.isnull().sum())
#Creating a statistical summary for the data
print()
print(data.describe())

#Splitting the dataset for weekend and weekday trends
data_W = data[data['Day Category'] == 'Weekend']
data_WY = data[data['Day Category'] == 'Weekday']


#EDA - Univariate

sns.barplot(data, x= 'Season_', y= 'atemp', ci = False, palette= 'bright')
plt.xlabel('')
plt.ylabel('Feeling Temperature')
plt.title('Seasons and their Average Temperature')
plt.show()

sns.histplot(data['count'])
plt.title("Distribution of total ride count")
plt.show()

#Subplot describing difference between casual and registered ridership

plt.subplot(2, 1, 1)
sns.histplot(data["registered"], color = 'red', kde= True)
plt.ylabel('Volume')
plt.xlabel('')
plt.title('Registered (Red) V. Casual (Green) Ride Count Distribution')
#plt.title('Casual Ridership')

plt.subplot(2, 1, 2)
sns.histplot(data["casual"], color = 'green', kde= True)
plt.ylabel('Volume')
plt.xlabel('Ride Count')
plt.show()

#Ride Count Distribution by rider type

fig, axes = plt.subplots(nrows = 2, ncols = 1, sharex= True)
sns.boxplot(x = data['registered'], color = 'red', ax = axes[0])
axes[0].set_title('Ride Distribution for Registered (Red) vs. Casual (Green) Users')
axes[0].tick_params(axis='x', which='both', labelbottom=True)

sns.boxplot(x= data['casual'], color= 'green', ax = axes[1])
axes[1].set_xlabel('Ride Count')
axes[1].set_ylabel('')
plt.show()

#Ride Count according to seasonality

soft_green = '#A8D08D'
sun_yellow = '#F1E0A6'
Earthy =  '#D97C3C'
ICE_BLUE = '#A0D0D9' #Create customer colours to reflect the respective seasons

sns.barplot(x= data['Season_'], y = data['count'], ci = False, palette= [soft_green, sun_yellow, Earthy, ICE_BLUE])
plt.title("Average Daily Ride Count by Season")
plt.ylabel("Daily Ride Count")
plt.xlabel('')
plt.show()

#Day of the week and rider count
sns.boxplot(x =data['count'], y = data['Day of the week'], order= ['Sun', 'Mon', 'Tue', 'Wed', 'Thu', 'Fri', 'Sat'], palette = 'bright')
plt.xlabel('')
plt.ylabel('')
plt.title('Ridership Distribution by Day of the Week')
plt.show()

#TIme of the day and ridership
# sns.lineplot(data, x= 'Time of day', y = 'count', ci = False)
# plt.xlabel('')
# plt.ylabel('Ride Count')
# plt.xticks(rotation = 45)
# plt.title('Ride Count Across The Day')

#Weekday V Weekday time of the day and ridership
plt.subplot(2, 1, 1)
sns.lineplot(data_WY, x= 'Time of day', y = 'count', color = 'red', ci = False)
plt.xlabel('')
plt.xticks(rotation = 45, fontsize = 7.8)
plt.ylabel('Ride Count')
plt.title('Ride Count Across the Day - Weekdays (Red) v. Weekends (Green)')

plt.subplot(2, 1, 2)
sns.lineplot(data_W, x= 'Time of day', y= 'count', ci= False, color= 'green')
plt.xlabel('Time of the Day')
plt.xticks(rotation = 45, fontsize = 7.8)
plt.ylabel('Ride Count')
plt.show()

#Correlation Heatmap

DN = data[['holiday',    'workingday',       'weather',         'temp',         'atemp',      'humidity',     'windspeed',  'count']]

Corr_Matrix = DN.corr()
sns.heatmap(Corr_Matrix, annot= True )
plt.xticks(rotation = 45)
plt.show()

#VIF

import statsmodels
from statsmodels.stats.outliers_influence import variance_inflation_factor

VIF_data = pd.DataFrame()

VIF_data['Feature'] = DN.columns
VIF_data['VIF'] = [variance_inflation_factor(DN.values, i) for i in range(len(DN.columns))]
print('VIF\n',VIF_data)
from sklearn.decomposition import PCA

#Dimensionality reduction - Create Copy of dataa

#Create a model version of the dataset for the model

DNM = DN[['holiday', 'workingday', 'weather', 'atemp', 'count' ]]

print(DNM.head())

#Building model

import sklearn
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import  train_test_split
from sklearn.linear_model import LinearRegression


scaler = StandardScaler()

scaled = scaler.fit_transform(DNM)

#Convert back into dataframe

DATA_M = pd.DataFrame(scaled, columns= DNM.columns.tolist())

print(DATA_M.head())

#Create Dependent and Independent variables

y = DATA_M['count']

X = DATA_M[['holiday', 'workingday', 'atemp', 'weather']]

#Training the model

X_train, X_test, y_train, y_test = train_test_split(X, y, random_state= 11, test_size= 0.2)

LR = LinearRegression()

LR.fit(X_train, y_train)

print(LR.coef_)

coefs = pd.DataFrame(LR.coef_, X.columns, columns= ['Coefficients'])

print(coefs)
