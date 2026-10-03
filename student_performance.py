# Student Performance Prediction
# Machine Learning Project - Module 6

import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, r2_score


# Create a sample student dataset
data = {
    "study_hours": [1, 2, 2, 3, 3, 4, 4, 5, 5, 6,
                    6, 7, 7, 8, 8, 9, 9, 10, 10, 11],
    "attendance": [60, 65, 68, 70, 72, 75, 78, 80, 82, 84,
                   85, 87, 88, 90, 91, 92, 93, 95, 96, 98],
    "previous_score": [45, 50, 52, 55, 58, 60, 62, 65, 67, 70,
                       71, 73, 75, 78, 80, 82, 84, 87, 90, 92],
    "final_score": [48, 52, 55, 58, 61, 64, 66, 69, 72, 75,
                    77, 79, 81, 84, 86, 88, 90, 93, 95, 97]
}

df = pd.DataFrame(data)

print("Student Performance Dataset")
print(df.head())


# Features and target
X = df[["study_hours", "attendance", "previous_score"]]
y = df["final_score"]


# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)


# Create and train the ML model
model = LinearRegression()
model.fit(X_train, y_train)


# Make predictions
y_pred = model.predict(X_test)


# Evaluate the model
mae = mean_absolute_error(y_test, y_pred)
r2 = r2_score(y_test, y_pred)

print("\nModel Evaluation")
print("Mean Absolute Error:", round(mae, 2))
print("R2 Score:", round(r2, 2))


# Predict performance of a new student
new_student = pd.DataFrame(
    [[7, 90, 80]],
    columns=["study_hours", "attendance", "previous_score"]
)

prediction = model.predict(new_student)

print("\nNew Student Prediction")
print("Study Hours: 7")
print("Attendance: 90%")
print("Previous Score: 80")
print("Predicted Final Score:", round(prediction[0], 2))