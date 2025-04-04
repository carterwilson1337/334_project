import kagglehub
import pandas as pd

from sklearn import linear_model
from sklearn import preprocessing

# Download latest version
path = kagglehub.dataset_download("adilshamim8/sleep-cycle-and-productivity")

print("Path to dataset files:", path)

df = pd.read_csv(path + "/sleep_cycle_productivity.csv")


#Cleaning the data





X = df[['Age', 'Sleep Start Time',
       'Sleep End Time', 'Total Sleep Hours', 'Sleep Quality',
       'Exercise (mins/day)', 'Caffeine Intake (mg)',
       'Screen Time Before Bed (mins)', 'Work Hours (hrs/day)', 'Mood Score', 'Stress Level']]


#Normalize data
scaler = preprocessing.StandardScaler().fit(X)

#Save columns for printing later
cols = X.columns
X = scaler.transform(X)
y = df[['Productivity Score']]


regr = linear_model.LinearRegression()
regr.fit(X, y)

print(regr.coef_)
for i in range(len(regr.coef_[0])):
    print(str(cols[i]) + ": " + str(regr.coef_[0][i]))

print(df['Sleep End Time'])