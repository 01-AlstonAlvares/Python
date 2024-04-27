import numpy as np
import seaborn as sns
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score
from sklearn.preprocessing import LabelEncoder

import pandas as pd
import random

data = {'student_id': [random.randint(1, 50) for _ in range(100)],
        'interaction_type': [random.choice(['click', 'hover', 'watch']) for _ in range(100)],
        'duration': [random.uniform(1, 10) for _ in range(100)],
        'timestamp': [pd.Timestamp('2024-04-27') + pd.Timedelta(seconds=i) for i in range(100)],
        'resource_id': [random.randint(1, 10) for _ in range(100)]}

df = pd.DataFrame(data)
df.to_csv('student_interaction_data.csv', index=False)


class LearningSystem:
    def __init__(self, data):
        self.data = data
        self.features = data.columns[:-1]
        self.target = data.columns[-1]

    def preprocess(self):
        # Implement data preprocessing steps such as missing value imputation, feature scaling, etc.
        pass

    def train_model(self):
        # One-hot encode the interaction_type column
        X = pd.get_dummies(self.data[['student_id', 'duration', 'timestamp', 'resource_id']])
        y = self.data['interaction_type']

        # Convert the categorical values to numerical codes
        le = LabelEncoder()
        y = le.fit_transform(y)

        # Split the data into training and testing sets
        X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

        # Train the linear regression model
        model = LinearRegression()
        model.fit(X_train.values, y_train)

        # Evaluate the model on the testing set
        y_pred = model.predict(X_test.values)
        mse = mean_squared_error(y_test, y_pred)
        r2 = r2_score(y_test, y_pred)
        print(f"MSE: {mse}, R2: {r2}")

    def analyze_cognitive_engagement(self):
        # Implement cognitive engagement analysis using the ICAP framework
        pass


data = pd.read_csv("student_interaction_data.csv")
system = LearningSystem(data)
system.preprocess()
system.train_model()
system.analyze_cognitive_engagement()
