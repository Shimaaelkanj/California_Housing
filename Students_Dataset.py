import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score
import numpy as np



data = {
    'Study_Hours':      [7, 2, 8, 3, 6, 1, 5, 4, 9, 3, 10, 5, 4, 7, 2, 8, 5, 9, 1, 6],
    'Previous_Grade':   [3, 1, 3, 2, 2, 1, 2, 3, 3, 1, 3, 2, 1, 3, 1, 2, 3, 3, 1, 2],
    'Attendance':       [1, 0, 1, 0, 1, 0, 0, 0, 1, 1, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1],
    'Pass_Fail':        [1, 0, 1, 0, 1, 0, 1, 0, 1, 0, 1, 1, 0, 1, 0, 1, 1, 1, 0, 1]
}


df = pd.DataFrame(data)



print("=== Student Dataset Preview ===")
print(df.head(10))
print("-" * 50)

print("=== All Records ===")
print(df)
print("-" * 50)


X = df[['Study_Hours', 'Previous_Grade', 'Attendance']]
y = df['Pass_Fail']


X_train, X_test, y_train, y_test = train_test_split(
    X, y,
    test_size=0.3,
    random_state=42,
    stratify=y
)

print(f"Training set size: {len(X_train)} records")
print(f"Testing set size: {len(X_test)} records")
print("-" * 50)



error_rates = []

for k in range(1, len(X_train)):
    knn = KNeighborsClassifier(n_neighbors=k)
    knn.fit(X_train, y_train)

    y_pred_k = knn.predict(X_test)

    error = np.mean(y_pred_k != y_test)
    error_rates.append(error)

best_k = np.argmin(error_rates) + 1

print("Error Rates for different K values:")
for i, error in enumerate(error_rates):
    print(f"K={i+1} -> Error Rate={error:.2f}")

print(f"\nOptimal K value: {best_k}")
print("-" * 50)



final_k = best_k


if best_k == 1:
    final_k = 3
    print(f"Using K={final_k} instead of 1 for better generalization")
else:
    print(f"Using optimal K={final_k}")

knn_classifier = KNeighborsClassifier(n_neighbors=final_k)

knn_classifier.fit(X_train, y_train)



y_pred = knn_classifier.predict(X_test)

print("\n=== MODEL EVALUATION ===")


accuracy = accuracy_score(y_test, y_pred)
print(f"Accuracy: {accuracy*100:.2f}%")


print("\nConfusion Matrix:")
cm = confusion_matrix(y_test, y_pred)
print(cm)


print("\nClassification Report:")
print(classification_report(
    y_test,
    y_pred,
    target_names=["Fail", "Pass"]
))


student_a = pd.DataFrame({
    'Study_Hours': [9],
    'Previous_Grade': [3],
    'Attendance': [1]
})

prediction_a = knn_classifier.predict(student_a)[0]

if prediction_a == 1:
    result = "Pass"
else:
    result = "Fail"

print("\n=== Prediction Example ===")
print("Student: Study_Hours=9, Grade=A, Attendance=High")
print(f"Prediction: {result}")


student_b = pd.DataFrame({
    'Study_Hours': [2],
    'Previous_Grade': [1],
    'Attendance': [0]
})

prediction_b = knn_classifier.predict(student_b)[0]

result_b = "Pass" if prediction_b == 1 else "Fail"

print("\nStudent: Study_Hours=2, Grade=C, Attendance=Low")
print(f"Prediction: {result_b}")

