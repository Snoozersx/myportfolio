# Regression visulzation script, Jared Gates

#The code first downloads a housing dataset from GitHub and loads it into a Pandas DataFrame. 
#It then explores the data, displaying the first 5 rows, information about the DataFrame, and summary statistics. 
#Next, it creates a scatter plot with a regression line using seaborn and prepares the data for modeling by splitting it into features and target. 
#It further splits the data into training and testing sets using scikit-learn's train_test_split function. 
#After fitting a linear regression model to the training data, it makes predictions on the testing set and calculates the mean squared error and R2 score for the model's predictions. 
#Finally, it prints these scores.



import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
import ssl
import urllib.request

# Disable SSL certificate verification
ctx = ssl.create_default_context()
ctx.check_hostname = False
ctx.verify_mode = ssl.CERT_NONE

url = "https://raw.githubusercontent.com/ageron/handson-ml/master/datasets/housing/housing.csv"
data = pd.read_csv(urllib.request.urlopen(url, context=ctx))

data.head()
data.info()
data.describe()

sns.pairplot(data, hue="ocean_proximity")
plt.show()

# Split the data into features (X) and target (y)
X = data.drop("median_house_value", axis=1)
y = data["median_house_value"]

# Split the data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

model = LinearRegression()
model.fit(X_train, y_train)

# Make predictions on the testing set
y_pred = model.predict(X_test)

# Calculate the mean squared error and R2 score
mse = mean_squared_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("Mean Squared Error:", mse)
print("R2 Score:", r2)