import pandas as pd
import matplotlib.pyplot as plt  
from sklearn.naive_bayes import GaussianNB
import numpy as np
from sklearn.metrics import mean_squared_error


# Step 1: Load Data from CSV File
dataframe = pd.read_csv('titanic.csv')
# print(dataframe.head())
dataframe = dataframe.drop(['Name'], axis=1)

# Step 2: Plot the Data
ages = dataframe['Age'].values
fares = dataframe['Fare'].values
survived = dataframe['Survived']
# values will convert the dataframe object into a numpy array
colors = []
for item in survived:
    if(item==0):
        colors.append('Red')
    else:
        colors.append('Green')
# plt.scatter(ages, fares, s=50, color=colors)
# s means size, we want size to be bigger
# plt.show()

# Step 3: Build a NB Model
Features = dataframe.drop(['Survived'], axis=1).values
Targets = dataframe['Survived'].values
Features_Train, Target_Train = Features[:710], Targets[:710]
# there are total 887 data points and 80% of that will be 710
Features_Test, Targets_test = Features[710:], Targets[710:]
# print(Features_Test)

model = GaussianNB()
model.fit(Features_Train, Target_Train)

# Step 4: Print Predicted vs Actuals
predicted_values = model.predict(Features_Test)
for item in zip(Targets_test, predicted_values):
    print('Actual was:', item[0], 'Predicted was', item[1])

# Step 5: Estimate Error
print('Accuracy is:', model.score(Features_Test, Targets_test))
# we didnt gave targets_test and predicted_values because
# score method itself calculates the predicted values from features_test and compares it with
# target_test and gives us the score
