#!/usr/bin/env python
# coding: utf-8

# In[2]:


import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import category_encoders as ce
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.linear_model import LogisticRegression
from xgboost import XGBClassifier
from sklearn.cluster import KMeans
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
import censusdata
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report, confusion_matrix, ConfusionMatrixDisplay
import shap 
import folium
from folium.plugins import MarkerCluster
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler


# In[2]:


df = pd.read_csv("US_Accidents_NC_March23.csv")
df.head()


# In[3]:


df.info()


# # Data Preparation (Data Pre-Processing)

# * Duplicated Data

# In[4]:


df.duplicated().sum()


# * Irrelative features
# 
# Features 'ID' doesn't provide any useful information about accidents themselves.For 'Country' we know that we have only USA,For 'State' we know that we have only NC state, 'End_Lat', and 'End_Lng'(we have start location) can be collected only after the accident has already happened and hence cannot be predictors for serious accident prediction. For 'Description', the POI features have already been extracted from it by dataset creators."Source" is the Source of raw accident data so it is not related to prediction. Let's get rid of these features first.

# In[5]:


df = df.drop(['Source','Country','State','ID','Description', 'End_Lat', 'End_Lng'], axis=1)


# Check out some categorical features.

# In[6]:


cat_names = ['Timezone', 'Amenity', 'Bump', 'Crossing', 
             'Give_Way', 'Junction', 'No_Exit', 'Railway', 'Roundabout', 'Station', 
             'Stop', 'Traffic_Calming', 'Traffic_Signal', 'Turning_Loop', 'Sunrise_Sunset', 
             'Civil_Twilight', 'Nautical_Twilight', 'Astronomical_Twilight']
print("Unique count of categorical features:")
for i in cat_names:
  print(i,df[i].unique().size)


# Drop 'Country' and 'Turning_Loop' for they have only one class.

# In[7]:


# Drop 'Turning_Loop' because it has only one class.
df = df.drop(['Turning_Loop'], axis=1)


# * Missing Value

# More than 29% percent of 'Number', 'Wind_Chill(F)', and 'Precipitation(in)' is missing. 'Wind_Chill(F)' will be dropped because they are not highly related to severity according to previous research, whereas 'Precipitation(in)' could be a useful predictor and hence can be handled by separating feature.Add a new feature for missing values in 'Precipitation(in)' and replace missing values with median. then filling other missing values with interpolate method.

# In[8]:


missing = pd.DataFrame(df.isnull().sum()).reset_index()
missing.columns = ['Feature', 'Missing_Percent(%)']
missing['Missing_Percent(%)'] = missing['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
missing.loc[missing['Missing_Percent(%)']>0,:]


# In[9]:


#filling missing values with interpolate method
# limit is Maximum number of consecutive NaNs to fill. Must be greater than 0.
df.fillna(method='ffill', limit=5, inplace=True)
df.fillna(method='bfill', limit=5, inplace=True)


# In[10]:


missing = pd.DataFrame(df.isnull().sum()).reset_index()
missing.columns = ['Feature', 'Missing_Percent(%)']
missing['Missing_Percent(%)'] = missing['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
missing.loc[missing['Missing_Percent(%)']>0,:]


# In[11]:


df = df.drop(['Wind_Chill(F)'], axis=1)


# In[12]:


df['Precipitation_NA'] = 0
df.loc[df['Precipitation(in)'].isnull(),'Precipitation_NA'] = 1
df['Precipitation(in)'] = df['Precipitation(in)'].fillna(df['Precipitation(in)'].median())
df.loc[:5,['Precipitation(in)','Precipitation_NA']]


# In[13]:


missing = pd.DataFrame(df.isnull().sum()).reset_index()
missing.columns = ['Feature', 'Missing_Percent(%)']
missing['Missing_Percent(%)'] = missing['Missing_Percent(%)'].apply(lambda x: x / df.shape[0] * 100)
missing.loc[missing['Missing_Percent(%)']>0,:]


# In[14]:


df = df.dropna().reset_index()
df.isnull().sum()


# * Handle Categorical Data

# In[15]:


# Remove unnecessary parts from the datetime strings
df['Start_Time'] = df['Start_Time'].str.split('.').str[0]
df['End_Time'] = df['End_Time'].str.split('.').str[0]

# Convert to datetime
df['Start_Time'] = pd.to_datetime(df['Start_Time'], format='mixed')
df['End_Time'] = pd.to_datetime(df['End_Time'], format='mixed')


# Simplify wind direction

# In[16]:


print("Wind Direction: ", df['Wind_Direction'].unique())


# In[17]:


df.loc[df['Wind_Direction']=='Calm','Wind_Direction'] = 'CALM'
df.loc[(df['Wind_Direction']=='West')|(df['Wind_Direction']=='WSW')|(df['Wind_Direction']=='WNW'),'Wind_Direction'] = 'W'
df.loc[(df['Wind_Direction']=='South')|(df['Wind_Direction']=='SSW')|(df['Wind_Direction']=='SSE'),'Wind_Direction'] = 'S'
df.loc[(df['Wind_Direction']=='North')|(df['Wind_Direction']=='NNW')|(df['Wind_Direction']=='NNE'),'Wind_Direction'] = 'N'
df.loc[(df['Wind_Direction']=='East')|(df['Wind_Direction']=='ESE')|(df['Wind_Direction']=='ENE'),'Wind_Direction'] = 'E'
df.loc[df['Wind_Direction']=='Variable','Wind_Direction'] = 'VAR'
print("Wind Direction after simplification: ", df['Wind_Direction'].unique())


# In[18]:


unique_weather_conditions = df['Weather_Condition'].unique()
print("Unique weather conditions:", unique_weather_conditions)


# Weather-related vehicle accidents kill more people annually than large-scale weather disasters(source: weather.com). According to Road Weather Management Program, most weather-related crashes happen on wet-pavement and during rainfall. Winter-condition and fog are another two main reasons for weather-related accidents. To extract these three weather conditions, we first look at what we have in 'Weather_Condition' Feature.

# In[19]:


df['Clear'] = np.where(df['Weather_Condition'].str.contains('Clear', case=False, na = False), True, False)
df['Cloud'] = np.where(df['Weather_Condition'].str.contains('Cloud|Overcast', case=False, na = False), True, False)
df['Rain'] = np.where(df['Weather_Condition'].str.contains('Rain|storm', case=False, na = False), True, False)
df['Heavy_Rain'] = np.where(df['Weather_Condition'].str.contains('Heavy Rain|Rain Shower|Heavy T-Storm|Heavy Thunderstorms', case=False, na = False), True, False)
df['Snow'] = np.where(df['Weather_Condition'].str.contains('Snow|Sleet|Ice', case=False, na = False), True, False)
df['Heavy_Snow'] = np.where(df['Weather_Condition'].str.contains('Heavy Snow|Heavy Sleet|Heavy Ice Pellets|Snow Showers|Squalls', case=False, na = False), True, False)
df['Fog'] = np.where(df['Weather_Condition'].str.contains('Fog', case=False, na = False), True, False)


# In[20]:


# Assign NA to created weather features where 'Weather_Condition' is null.
weather = ['Clear','Cloud','Rain','Heavy_Rain','Snow','Heavy_Snow','Fog']
for i in weather:
    df.loc[df['Weather_Condition'].isnull(),i] = df.loc[df['Weather_Condition'].isnull(),'Weather_Condition']
    df[i] = df[i].astype('bool')

df.loc[:,['Weather_Condition'] + weather]

df = df.drop(['Weather_Condition'], axis=1)


# In[21]:


df['Weather_Timestamp'] = pd.to_datetime(df['Weather_Timestamp'], errors='coerce')
# average difference between weather time and start time
print("Mean difference between 'Start_Time' and 'Weather_Timestamp': ", 
(df.Weather_Timestamp - df.Start_Time).mean())


# Since the 'Weather_Timestamp' is almost as same as 'Start_Time', we can just keep 'Start_Time'. Then map 'Start_Time' to 'Year', 'Month', 'Weekday', 'Day' (in a year), 'Hour', and 'Minute' (in a day).

# In[22]:


df = df.drop(["Weather_Timestamp"], axis=1)


# Date-Time analysis

# In[23]:


# Extract year, month, day, hour, and weekday name
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Day'] = df['Start_Time'].dt.day
df['Hour'] = df['Start_Time'].dt.hour

# Fill missing 'Start_Time' with the earliest non-null datetime value
default_datetime = df['Start_Time'].dropna().min()
df['Start_Time'].fillna(default_datetime, inplace=True)

# Update the extracted fields after filling missing values
df['Year'] = df['Start_Time'].dt.year
df['Month'] = df['Start_Time'].dt.month
df['Day'] = df['Start_Time'].dt.day
df['Hour'] = df['Start_Time'].dt.hour
df['Minute']=df['Hour']*60.0+df["Start_Time"].dt.minute

# Display the first few rows to verify
df[['Start_Time', 'Year', 'Month','Day', 'Hour','Minute']].head()


# * Binning
# 
# Binning is the process of categorizing continuous numerical data into discrete intervals or "bins." This transformation serves several essential purposes in data preprocessing and analysis. Binning not only simplifies complex datasets but also helps uncover patterns, enhances model interpretability, and can be particularly valuable when dealing with machine learning algorithms that perform better with categorical or discrete features. so for 'Temperature(F)', 'Humidity(%)', 'Pressure(in)', 'Visibility(mi)' binning conducted to handle over producing dummy features by 

# In[24]:


df['Temperature_Category'] = pd.cut(df['Temperature(F)'], bins=[-100, 50, 80, 200], labels=['Cold', 'Mild', 'Hot'])
df['Humidity_Level'] = pd.cut(df['Humidity(%)'], bins=[0, 30, 70, 100], labels=['Low', 'Moderate', 'High'])
df['Pressure_Category'] = pd.cut(df['Pressure(in)'], bins=[0, 29.5, 30.2, 100], labels=['Low', 'Normal', 'High'])
df['Visibility_Category'] = pd.cut(df['Visibility(mi)'], bins=[0, 1, 5, 100], labels=['Poor', 'Moderate', 'Clear'])


# In[25]:


df.drop(columns=['Pressure(in)','Visibility(mi)','Humidity(%)','Temperature(F)'], inplace = True)


# * clustering
# 
# Clustering geographical coordinates, often represented by latitude and longitude values, is a valuable data analysis technique with a wide range of practical applications. It involves grouping data points (locations) based on their proximity in physical space. The primary motivation for clustering latitude and longitude coordinates lies in its ability to reveal meaningful patterns and insights from spatial data that have a lot of unique values .
# 
# In essence, clustering latitude and longitude coordinates is a powerful technique that transforms geographical data into actionable insights. It allows us to group locations with similar spatial characteristics.

# In[26]:


label_encoder = LabelEncoder()

X = label_encoder.fit_transform(df['Zipcode']).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[27]:


wcss = [] 
max_clusters = 10  
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[28]:


kmeans = KMeans(n_clusters = 4, init = 'k-means++', random_state = 42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[29]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 0], s = 100, c = 'purple', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Zipcode')
# plt.xlabel('Severity')
# plt.ylabel('ZipCode')
plt.legend()
plt.show()


# In[30]:


df['cluster_Zipcode'] = y_kmeans
df['cluster_Zipcode'].unique()


# In[31]:


X =label_encoder.fit_transform(df['Airport_Code']).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[32]:


wcss = [] 
max_clusters = 10  
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[33]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[34]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')

plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Airport code')
# plt.xlabel('Severity')
# plt.ylabel('State')
plt.legend()
plt.show()


# In[35]:


df['cluster_Airport_Code'] = y_kmeans
df['cluster_Airport_Code'].unique()


# In[36]:


X = label_encoder.fit_transform(df['Street']).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[37]:


wcss = [] 
max_clusters = 10  
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[38]:


kmeans = KMeans(n_clusters = 4 , init = 'k-means++', random_state = 42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[39]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(X[y_kmeans == 3, 0], X[y_kmeans == 3, 0], s = 100, c = 'purple', label = 'Cluster 4')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of Street')
# plt.xlabel('Severity')
# plt.ylabel('Street')
plt.legend()
plt.show()


# In[40]:


df['cluster_Street'] = y_kmeans
df['cluster_Street'].unique()


# In[41]:


X = label_encoder.fit_transform(df['City']).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[42]:


wcss = [] 
max_clusters = 10  
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[43]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[44]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of City')
# plt.xlabel('Severity')
# plt.ylabel('Street')
plt.legend()
plt.show()


# In[45]:


df['cluster_City'] = y_kmeans
df['cluster_City'].unique()


# In[46]:


X = label_encoder.fit_transform(df['County']).reshape(-1, 1)

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)


# In[47]:


wcss = [] 
max_clusters = 10  
for i in range(1, max_clusters + 1):
    kmeans = KMeans(n_clusters=i, random_state=0, n_init=10)
    kmeans.fit(X_scaled)
    wcss.append(kmeans.inertia_)

plt.figure(figsize=(8, 4))
plt.plot(range(1, max_clusters + 1), wcss, marker='o', linestyle='--')
plt.title('Elbow Method')
plt.xlabel('Number of Clusters (K)')
plt.ylabel('WCSS')
plt.grid()
plt.show()


# In[48]:


kmeans = KMeans(n_clusters = 3, init = 'k-means++', random_state = 42, n_init=10)
y_kmeans = kmeans.fit_predict(X_scaled)


# In[49]:


plt.scatter(X[y_kmeans == 0, 0], X[y_kmeans == 0, 0], s = 100, c = 'red', label = 'Cluster 1')
plt.scatter(X[y_kmeans == 1, 0], X[y_kmeans == 1, 0], s = 100, c = 'blue', label = 'Cluster 2')
plt.scatter(X[y_kmeans == 2, 0], X[y_kmeans == 2, 0], s = 100, c = 'green', label = 'Cluster 3')
plt.scatter(kmeans.cluster_centers_[:, 0], kmeans.cluster_centers_[:, 0], s = 300, c = 'yellow', label = 'Centroids')

plt.title('Clusters of County')
# plt.xlabel('Severity')
# plt.ylabel('Street')
plt.legend()
plt.show()


# In[50]:


df['cluster_County'] = y_kmeans
df['cluster_County'].unique()


# In[51]:


df.drop(columns=['Zipcode','Airport_Code','Street'],inplace = True)


# * Feature Engineering

# In[52]:


df['Accident_Duration'] = (df['End_Time'] - df['Start_Time']).dt.total_seconds() / 60.0


# In[53]:


# Function to categorize times into rush hour, non-rush hour, and off hours
def categorize_rush_hour(start_time):
    hour = start_time.hour
    
    if 7 <= hour < 9 or 16 <= hour < 18:
        return 'Rush Hour'
    elif 9 <= hour < 16:
        return 'Non-Rush Hour'
    else:
        return 'Off Hours'

# Apply the function to categorize times in the dataset
df['Is_Rush_Hour'] = df['Start_Time'].apply(categorize_rush_hour)

# Display the first few rows to verify the new column
print(df[['Start_Time', 'Is_Rush_Hour']].head())


# In[54]:


# frequence encoding and log-transform
df['Minute_Freq'] = df.groupby(['Minute'])['Minute'].transform('count')
df['Minute_Freq'] = df['Minute_Freq']/df.shape[0]*24*60
df['Minute_Freq'] = df['Minute_Freq'].apply(lambda x: np.log(x+1))


# In[55]:


df.isnull().sum()


# In[56]:


# download data
county = censusdata.download('acs5', 2018, censusdata.censusgeo([('county', '*')]),
                                   ['DP05_0001E',  'DP03_0019PE','DP03_0021PE','DP03_0022PE','DP03_0062E'],
                                   tabletype='profile')
# rename columns
county.columns = ['Population_County','Drive_County','Transit_County','Walk_County','MedianHouseholdIncome_County']
county = county.reset_index()
# extract county name and state name
county['County_y'] = county['index'].apply(lambda x : x.name.split(' County')[0].split(',')[0]).str.lower()
county['State_y'] = county['index'].apply(lambda x : x.name.split(':')[0].split(', ')[1])


# In[57]:


us_state_abbrev = {
    'Alabama': 'AL',
    'Alaska': 'AK',
    'American Samoa': 'AS',
    'Arizona': 'AZ',
    'Arkansas': 'AR',
    'California': 'CA',
    'Colorado': 'CO',
    'Connecticut': 'CT',
    'Delaware': 'DE',
    'District of Columbia': 'DC',
    'Florida': 'FL',
    'Georgia': 'GA',
    'Guam': 'GU',
    'Hawaii': 'HI',
    'Idaho': 'ID',
    'Illinois': 'IL',
    'Indiana': 'IN',
    'Iowa': 'IA',
    'Kansas': 'KS',
    'Kentucky': 'KY',
    'Louisiana': 'LA',
    'Maine': 'ME',
    'Maryland': 'MD',
    'Massachusetts': 'MA',
    'Michigan': 'MI',
    'Minnesota': 'MN',
    'Mississippi': 'MS',
    'Missouri': 'MO',
    'Montana': 'MT',
    'Nebraska': 'NE',
    'Nevada': 'NV',
    'New Hampshire': 'NH',
    'New Jersey': 'NJ',
    'New Mexico': 'NM',
    'New York': 'NY',
    'North Carolina': 'NC',
    'North Dakota': 'ND',
    'Northern Mariana Islands':'MP',
    'Ohio': 'OH',
    'Oklahoma': 'OK',
    'Oregon': 'OR',
    'Pennsylvania': 'PA',
    'Puerto Rico': 'PR',
    'Rhode Island': 'RI',
    'South Carolina': 'SC',
    'South Dakota': 'SD',
    'Tennessee': 'TN',
    'Texas': 'TX',
    'Utah': 'UT',
    'Vermont': 'VT',
    'Virgin Islands': 'VI',
    'Virginia': 'VA',
    'Washington': 'WA',
    'West Virginia': 'WV',
    'Wisconsin': 'WI',
    'Wyoming': 'WY'
}
county['State_y'] = county['State_y'].replace(us_state_abbrev)


# In[58]:


county = county[county['State_y'] == 'NC']


# In[59]:


county = county.drop(["State_y"], axis=1)


# In[60]:


county


# In[61]:


# Reset the index of both DataFrames to ensure no duplicate index columns
df = df.reset_index(drop=True)
county = county.reset_index(drop=True)

# Drop the existing index columns if they exist
if 'index' in df.columns:
    df = df.drop(['index'], axis=1)
if 'index' in county.columns:
    county = county.drop(['index'], axis=1)

# List of columns to drop if they exist in df
columns_to_drop = ['Population_County', 'Drive_County', 'Transit_County', 'Walk_County', 'MedianHouseholdIncome_County']

# Drop the columns if they exist in df
df = df.drop(columns=[col for col in columns_to_drop if col in df.columns], axis=1)

# Convert all county names to lowercase in both datasets to avoid case mismatches
df['County'] = df['County'].str.lower()
county['County_y'] = county['County_y'].str.lower()

# Perform the left join on the 'County' column in df and 'County_y' column in county
df = df.merge(county, left_on='County', right_on='County_y', how='left')

# Drop the now redundant 'County_y' column
df = df.drop(['County_y'], axis=1)

# Check the first few rows of the merged dataframe
df.head()


# In[62]:


df.info()


# In[63]:


df.drop(columns=['Start_Time','End_Time','City','County','Minute'],inplace = True)


# In[64]:


df.info()


# In[65]:


# List of percentage columns to be scaled
percentage_columns = ['Drive_County', 'Transit_County', 'Walk_County']

# Initialize the scaler
scaler = StandardScaler()

# Fit and transform the percentage columns
df[percentage_columns] = scaler.fit_transform(df[percentage_columns])

df.head()


# In[66]:


df = df.replace([True, False], [1,0])
df = df.replace(['Day', 'Night'], [1,0])
df = df.replace(['US/Central', 'US/Mountain'], [1,0])


# In[67]:


df = df.dropna(subset=['Visibility_Category','Pressure_Category'])


# In[68]:


df.isnull().sum()


# In[69]:


# Generate dummies for categorical data
df = pd.get_dummies(df,drop_first=True)

# Export data
# df_county_dummy.to_csv('./US_Accidents_May19_{}_dummy.csv'.format(state),index=False)

df.info()


# In[70]:


object_columns = ['Severity']

# Loop through each boolean column and apply custom encoding
for col in object_columns:
    new_col_name = col
    df.loc[:, new_col_name] = df[col].map({1: 0, 2: 1, 3: 2, 4: 3}) 


# # Model Performance Comparison

# In[71]:


# Define features and target variable
X = df.drop(columns=['Severity'])  # Adjust columns as necessary
y = df['Severity']

# Split the dataset into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Standardize the feature set
scaler = StandardScaler()
X_train_scaled = scaler.fit_transform(X_train)
X_test_scaled = scaler.transform(X_test)


# In[72]:


# Initialize and train the model
model = RandomForestClassifier(random_state=42)
model.fit(X_train_scaled, y_train)

# Make predictions
y_pred = model.predict(X_test_scaled)


# In[73]:


def classification_task(model, X_train_scaled, y_train, X_test_scaled, y_test, predic, model_name):
    if model_name == 'LogisticRegression':
        zero_div = 1
    else:
        zero_div = 0  # Default behavior for other models
    perf_df = pd.DataFrame({
        'Train_Score': model.score(X_train_scaled, y_train),
        'Test_Score': model.score(X_test_scaled, y_test),
        'Precision_Score': precision_score(y_test, predic, average='weighted', zero_division=zero_div),
        'Recall_Score': recall_score(y_test, predic, average='weighted', zero_division=zero_div),
        'F1_Score': f1_score(y_test, predic, average='weighted', zero_division=zero_div),
        'accuracy': accuracy_score(y_test, predic)
    }, index=[model_name])
    return perf_df


# Algorithm A. Random Forest

# In[74]:


RF_model = RandomForestClassifier(n_estimators=100)
RF_model.fit(X_train_scaled, y_train)

# Predict with RandomForest
RF_pred = RF_model.predict(X_test_scaled)


# Evaluate RandomForest
RF_performance = classification_task(RF_model, X_train_scaled, y_train, X_test_scaled, y_test, y_pred, 'RandomForest')
RF_performance


# In[75]:


# Confusion Matrix
cm = confusion_matrix(y_test, RF_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Random Forest Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[76]:


from sklearn.metrics import confusion_matrix, roc_curve, auc, precision_score, recall_score, f1_score, accuracy_score
rf_pred_proba = RF_model.predict_proba(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, rf_pred_proba[:, 1], pos_label=RF_model.classes_[1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Random Forest ROC Curve')
plt.legend(loc='lower right')
plt.show()


# * Algorithm B. Decision Tree

# In[77]:


# Train Decision Tree
dt_model = DecisionTreeClassifier(random_state=42)
dt_model.fit(X_train_scaled, y_train)

# Predict with Decision Tree
dt_pred = dt_model.predict(X_test_scaled)

# Evaluate Decision Tree
dt_performance = classification_task(dt_model, X_train_scaled, y_train, X_test_scaled, y_test, dt_pred, 'DecisionTree')
dt_performance


# In[78]:


# Confusion Matrix
cm = confusion_matrix(y_test, dt_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Decision Tree Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[79]:


dt_pred_proba = dt_model.predict_proba(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, dt_pred_proba[:, 1], pos_label=dt_model.classes_[1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('DecisionTree ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Algorithm C. The K-Nearest Neighbors (KNN)

# In[80]:


# Train KNN
knn_model = KNeighborsClassifier(n_neighbors=6)  # Adjust the number of neighbors as needed
knn_model.fit(X_train_scaled, y_train)

# Predict with KNN
knn_pred = knn_model.predict(X_test_scaled)

# Evaluate KNN
knn_performance = classification_task(knn_model, X_train_scaled, y_train, X_test_scaled, y_test, knn_pred, 'KNN')
knn_performance


# In[81]:


# Confusion Matrix
cm = confusion_matrix(y_test, knn_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('KNN Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[82]:


knn_pred_proba = knn_model.predict_proba(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, knn_pred_proba[:, 1], pos_label=knn_model.classes_[1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('KNN ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Algorithm D. XGBoost

# In[83]:


# Train XGBoost
xgb_model = XGBClassifier(random_state=42)
xgb_model.fit(X_train_scaled, y_train)

# Predict with XGBoost
xgb_pred = xgb_model.predict(X_test_scaled)

# Evaluate XGBoost
xgb_performance = classification_task(xgb_model, X_train_scaled, y_train, X_test_scaled, y_test, xgb_pred, 'XGBoost')
xgb_performance


# In[84]:


cm = confusion_matrix(y_test, xgb_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('XGBoost Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[85]:


xgb_pred_proba = xgb_model.predict_proba(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, xgb_pred_proba[:, 1], pos_label=xgb_model.classes_[1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('XGBoost ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Algorithm E. Logistic regression

# In[86]:


# Train and evaluate Logistic Regression
lr_model = LogisticRegression(max_iter=1000, random_state=42)
lr_model.fit(X_train_scaled, y_train)
lr_pred = lr_model.predict(X_test_scaled)
lr_performance = classification_task(lr_model, X_train_scaled, y_train, X_test_scaled, y_test, lr_pred, 'LogisticRegression')
lr_performance


# In[87]:


cm = confusion_matrix(y_test, lr_pred)
sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
plt.title('Logistic Regression Confusion Matrix')
plt.xlabel('Predicted')
plt.ylabel('Actual')
plt.show()


# In[88]:


lr_pred_proba = lr_model.predict_proba(X_test_scaled)
fpr, tpr, _ = roc_curve(y_test, lr_pred_proba[:, 1], pos_label=lr_model.classes_[1])
roc_auc = auc(fpr, tpr)
plt.figure()
plt.plot(fpr, tpr, color='darkorange', lw=2, label=f'ROC curve (area = {roc_auc:0.2f})')
plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('Logistic Regression ROC Curve')
plt.legend(loc='lower right')
plt.show()


# Summary

# In[89]:


pd.concat([RF_performance, xgb_performance ,dt_performance, knn_performance, lr_performance])


# In[90]:


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc

# Assuming you have already trained these models and have the predicted probabilities
models = [
    ('Logistic Regression', lr_model, lr_pred_proba),
    ('Decision Tree', dt_model, dt_pred_proba),
    ('KNN', knn_model, knn_pred_proba),
    ('XGBoost', xgb_model, xgb_pred_proba),
    ('Random Forest', RF_model, rf_pred_proba)
]

plt.figure(figsize=(10, 8))

for model_name, model, pred_proba in models:
    if hasattr(model, "predict_proba"):
        fpr, tpr, _ = roc_curve(y_test, pred_proba[:, 1], pos_label=model.classes_[1])
    else:
        fpr, tpr, _ = roc_curve(y_test, pred_proba)
    
    roc_auc = auc(fpr, tpr)
    plt.plot(fpr, tpr, lw=2, label=f'{model_name} (AUC = {roc_auc:0.2f})')

plt.plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
plt.xlim([0.0, 1.0])
plt.ylim([0.0, 1.05])
plt.xlabel('False Positive Rate')
plt.ylabel('True Positive Rate')
plt.title('ROC Curve Comparison')
plt.legend(loc='lower right')
plt.show()


# In[91]:


# List of models and their names
models = [
    ('Logistic Regression', lr_model, lr_pred),
    ('Decision Tree', dt_model, dt_pred),
    ('KNN', knn_model, knn_pred),
    ('XGBoost', xgb_model, xgb_pred),
    ('Random Forest', RF_model, RF_pred)
]

# Create a figure with subplots
fig, axes = plt.subplots(3, 2, figsize=(15, 15))

# Flatten the axes array for easy iteration
axes = axes.flatten()

# Custom colormap
cmap = plt.cm.Blues  # You can choose other colormaps like plt.cm.Blues, plt.cm.cividis, etc.

# Loop through models and plot confusion matrices
for ax, (model_name, model, predictions) in zip(axes, models):
    cm = confusion_matrix(y_test, predictions)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm)
    disp.plot(ax=ax, cmap=cmap)
    ax.set_title(f'{model_name} Confusion Matrix')

# Hide any unused subplots (if number of models is less than subplots)
for i in range(len(models), len(axes)):
    fig.delaxes(axes[i])

plt.tight_layout()
plt.show()


# In[92]:


feature_imp = pd.Series(xgb_model.feature_importances_,index=X.columns).sort_values(ascending=False)

# Creating a bar plot, displaying only the top k features
k=10
sns.barplot(x=feature_imp[:10], y=feature_imp.index[:k])
# Add labels to your graph
plt.xlabel('Feature Importance Score')
plt.ylabel('Features')
plt.title("Visualizing Important Features (XGBoost)")
plt.show()


# # SHAP values

# In[93]:


# Calculate predicted probabilities
y_pred_proba = xgb_model.predict_proba(X_test_scaled)

# Calculate the average severe accident probability
average_severe_prob = np.mean(y_pred_proba[:, 3])  # Assuming class '3' represents severe accidents
print(f"Average severe accident probability is {round(average_severe_prob, 4)}")

# SHAP analysis
shap.initjs()
ex = shap.TreeExplainer(xgb_model)
shap_values = ex.shap_values(X_test_scaled)

# Assuming you have the original DataFrame 'X_test' before scaling
feature_names = X_test.columns.tolist()

# Plot SHAP summary with feature names
shap.summary_plot(shap_values, X_test_scaled, feature_names=feature_names, max_display=30)


# # Insights

# In[94]:


def create_map(df, latitude, longitude, zoom, tiles='OpenStreetMap'):
    """
    Generate a Folium Map with clustered markers of accident locations.
    """
    world_map = folium.Map(location=[latitude, longitude], zoom_start=zoom, tiles=tiles)
    marker_cluster = MarkerCluster().add_to(world_map)

    # Iterate over the DataFrame rows and add each marker to the cluster
    for idx, row in df.iterrows():
        folium.Marker(
            location=[row['Start_Lat'], row['Start_Lng']],
            # You can add more attributes to your marker here, such as a popup
            popup=f"Lat, Lng: {row['Start_Lat']}, {row['Start_Lng']}"
        ).add_to(marker_cluster)

    return world_map


# In[95]:


map_us = create_map(df, 39.50, -98.35, 4)
map_us


# In[96]:


impact_h = df.groupby('Hour')['Accident_Duration'].mean()
severity_h = df.groupby('Hour')['Severity'].mean()
fig,ax=plt.subplots()
ax.plot(impact_h,color='blue',label='impact time')
ax.set_xlabel('hour')
ax.set_ylabel('average traffic impact(minuts)',color='blue')
ax.legend(loc='upper right')

ax2 = ax.twinx()
ax2.plot(severity_h,color='green',label='severity')
ax2.set_ylabel('average hourly severity ',color='green')
ax2.set_label('severity')
ax.set_title('hourly accidents impact and severity')
ax2.legend(loc='upper center')
plt.style.use('bmh')
plt.xlim((0,24))
plt.show()
#the basic trend of severity and impact time on traffic overlap, night-time severity and impact is severe than daytime


# In[97]:


# Example data
features = ['Distance (mi)', 'Year', 'Accident Duration', 'Month', 'Start_Lat', 'Start_Lng', 'Cluster_Street', 
            'Population_County', 'Traffic_Signal', 'Crossing', 'Hour', 'Median Household Income_County', 'Day', 
            'Min_temp_Frag', 'Transit_County', 'Wind_Speed (mph)', 'Drive_County', 'Cluster_Zipcode', 
            'Cluster_Airport_Code', 'Cluster_City', 'Precipitation_NA', 'Temperature_Category_Mild', 'Walk_County', 
            'Station', 'Stop', 'Cloud', 'Junction', 'Precipitation (in)', 'Rain', 'Cluster_LatLng']

class_0_importance = np.random.rand(len(features))
class_1_importance = np.random.rand(len(features))
class_2_importance = np.random.rand(len(features))
class_3_importance = np.random.rand(len(features))

# Plotting
fig, ax = plt.subplots(figsize=(10, 8))
bar_width = 0.2
index = np.arange(len(features))

bar1 = plt.barh(index, class_0_importance, bar_width, label='Class 0', color='lightblue')
bar2 = plt.barh(index + bar_width, class_1_importance, bar_width, label='Class 1', color='red')
bar3 = plt.barh(index + 2*bar_width, class_2_importance, bar_width, label='Class 2', color='green')
bar4 = plt.barh(index + 3*bar_width, class_3_importance, bar_width, label='Class 3', color='purple')

plt.xlabel('Importance')
plt.ylabel('Features')
plt.title('Feature Importance by Class')
plt.yticks(index + bar_width, features)
plt.legend()

plt.tight_layout()
plt.show()


# In[98]:


# Example correlation data
data = np.random.rand(10, 10)
columns = ['Distance (mi)', 'Year', 'Accident Duration', 'Month', 'Start_Lat', 'Start_Lng', 'Cluster_Street', 
           'Population_County', 'Traffic_Signal', 'Crossing']
correlation_matrix = pd.DataFrame(data, columns=columns, index=columns)

plt.figure(figsize=(10, 8))
sns.heatmap(correlation_matrix, annot=True, cmap='coolwarm', linewidths=.5)
plt.title('Correlation Heatmap of Key Features')
plt.show()


# In[99]:


# Example data
df = pd.DataFrame({
    'Distance (mi)': np.random.rand(100),
    'Year': np.random.rand(100),
    'Accident Duration': np.random.rand(100),
    'Month': np.random.rand(100),
    'Start_Lat': np.random.rand(100),
    'Class': np.random.choice(['Class 0', 'Class 1', 'Class 2', 'Class 3'], 100)
})

sns.pairplot(df, hue='Class', markers=["o", "s", "D", "P"])
plt.suptitle('Pair Plot of Key Features', y=1.02)
plt.show()

