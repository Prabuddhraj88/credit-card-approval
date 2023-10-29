# Import necessary libraries
import pandas as pd
import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report

# Step 1: Load the application and credit record datasets
print("Step 1: Loading datasets...")
application_record = pd.read_csv("application_record.csv")
credit_record = pd.read_csv("credit_record.csv")

# Step 2: Merging datasets on 'ID'
print("Step 2: Merging datasets...")
data = application_record.merge(credit_record, on='ID')

# Step 3: Data Exploration
print("Step 3: Exploring the data...")
# Display basic statistics of the dataset
print("Basic Statistics:")
print(data.describe())

# Show the first few rows of the merged dataset
print("First Few Rows:")
print(data.head())

# Step 4: Data Cleaning
print("Step 4: Cleaning the data...")
# Check for missing values
missing_values = data.isnull().sum()
print("Missing Values:")
print(missing_values)

# Remove rows with missing target variable
data = data.dropna(subset=['STATUS'])

# Step 5: Define a set of features for the model
print("Step 5: Defining features...")
features = [
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'CNT_CHILDREN', 'AMT_INCOME_TOTAL',
    'NAME_INCOME_TYPE', 'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE',
    'DAYS_BIRTH', 'DAYS_EMPLOYED', 'FLAG_MOBIL', 'FLAG_WORK_PHONE', 'FLAG_PHONE', 'FLAG_EMAIL',
    'OCCUPATION_TYPE', 'CNT_FAM_MEMBERS'
]

# Step 6: Select features and target variable
print("Step 6: Selecting features and target variable...")
X = data[features]
y = data['STATUS']

# Step 7: Convert categorical variables to numerical using one-hot encoding
print("Step 7: Converting categorical variables to numerical using one-hot encoding...")
X = pd.get_dummies(X, columns=[
    'CODE_GENDER', 'FLAG_OWN_CAR', 'FLAG_OWN_REALTY', 'NAME_INCOME_TYPE',
    'NAME_EDUCATION_TYPE', 'NAME_FAMILY_STATUS', 'NAME_HOUSING_TYPE', 'OCCUPATION_TYPE'
])

# Step 8: Data Visualization
print("Step 8: Visualizing the data...")
# Plot a histogram of income distribution
plt.figure(figsize=(10, 6))
sns.histplot(data['AMT_INCOME_TOTAL'], kde=True)
plt.title("Income Distribution")
plt.xlabel("Income")
plt.ylabel("Frequency")
plt.show()

# Step 9: Split the data into training and testing sets
print("Step 9: Splitting the data into training and testing sets...")
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 10: Initialize and train a Random Forest Classifier
print("Step 10: Initializing and training the Random Forest Classifier...")
rfc = RandomForestClassifier(n_estimators=100)
rfc.fit(X_train, y_train)

# Step 11: Make predictions on the test set
print("Step 11: Making predictions on the test set...")
y_pred = rfc.predict(X_test)

# Step 12: Evaluate the model using classification report
print("Step 12: Evaluating the model...")
report = classification_report(y_test, y_pred)

# Step 13: Display the classification report
print("Step 13: Displaying the classification report...")
print("Classification Report:\n", report)
