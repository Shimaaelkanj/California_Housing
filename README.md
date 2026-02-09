# California House Prices Prediction using Linear Regression

## 1. Project Overview

This project demonstrates how to build a Machine Learning model to predict house prices using the California Housing dataset. The model uses the **Linear Regression algorithm**, which is a supervised learning algorithm used for predicting continuous numerical values.

The project covers the complete Machine Learning pipeline, including:

* Loading the dataset
* Exploring and understanding the data
* Visualizing correlations between features
* Splitting the data into training and testing sets
* Training a Linear Regression model
* Making predictions
* Evaluating model performance

---

## 2. Dataset Description

The dataset used is the **California Housing dataset**, which is included in the Scikit-learn library.

Each row represents a district in California, and each column represents a feature.

### Features (Input Variables)

* **MedInc** – Median income in the district
* **HouseAge** – Median house age
* **AveRooms** – Average number of rooms
* **AveBedrms** – Average number of bedrooms
* **Population** – District population
* **AveOccup** – Average house occupancy
* **Latitude** – Latitude location
* **Longitude** – Longitude location

### Target (Output Variable)

* **MedHouseVal** – Median house value (this is what we predict)

---

## 3. Machine Learning Pipeline

The project follows a standard Machine Learning pipeline:

### Step 1: Import Libraries

Libraries used:

* pandas → data manipulation
* numpy → numerical computations
* matplotlib → plotting graphs
* seaborn → data visualization
* sklearn → machine learning tools

---

### Step 2: Load the Dataset

The dataset is loaded using:

```python
housing = fetch_california_housing(as_frame=True)
data = housing.frame
```

This creates a Pandas DataFrame containing all features and target values.

---

### Step 3: Data Exploration

We analyze the dataset using:

```python
data.info()
data.describe()
```

This helps us understand:

* Number of samples
* Number of features
* Data types
* Statistical summary (mean, min, max, etc.)

---

### Step 4: Data Visualization

We use a correlation heatmap:

```python
sns.heatmap(data.corr(), annot=True, cmap='coolwarm')
```

This shows relationships between features and the target variable.

For example:

* Median income has a strong positive correlation with house price.

---

### Step 5: Define Features and Target

We separate inputs (X) and output (y):

```python
X = data.drop('MedHouseVal', axis=1)
y = data['MedHouseVal']
```

* X = features
* y = target

---

### Step 6: Split the Dataset

We split the data into:

* 80% training data
* 20% testing data

```python
train_test_split(X, y, test_size=0.2, random_state=42)
```

Training data is used to train the model.
Testing data is used to evaluate performance.

---

### Step 7: Train the Linear Regression Model

We create and train the model:

```python
model = LinearRegression()
model.fit(X_train, y_train)
```

The model learns the relationship between features and house prices.

---

### Step 8: Make Predictions

We predict house prices using:

```python
y_pred = model.predict(X_test)
```

---

### Step 9: Evaluate the Model

We evaluate performance using three metrics:

#### Mean Squared Error (MSE)

Measures average squared prediction error.

Lower is better.

#### Root Mean Squared Error (RMSE)

Square root of MSE.

Represents prediction error in original units.

#### R² Score (R-squared)

Measures how well the model explains the data.

Values range from:

* 1 → Perfect model
* 0 → Poor model

---

## 4. Linear Regression Algorithm Explanation

Linear Regression finds the best linear relationship between inputs and output.

Mathematical form:

```
y = b0 + b1x1 + b2x2 + ... + bn*xn
```

Where:

* y = predicted house price
* x = features
* b = coefficients learned by the model

The goal is to minimize prediction error.

---

## 5. Project Output

The program produces:

1. Dataset information
2. Statistical summary
3. Correlation heatmap
4. Model evaluation metrics:

Example:

```
Mean Squared Error (MSE): 0.53
Root Mean Squared Error (RMSE): 0.73
R-squared (R2) Score: 0.61
```

This means the model explains about 61% of the variance in house prices.

---

## 6. Project Structure

```
ML projects/
│
└── California_House_Prices/
    │
    ├── California_House_Prices.py
    ├── README.md
    └── venv/
```

---

## 7. Requirements

Install required libraries:

```
pip install pandas numpy matplotlib seaborn scikit-learn
```

---

## 8. How to Run the Project

Activate virtual environment:

```
venv\Scripts\activate
```

Run the script:

```
python California_House_Prices.py
```

---

## 9. Conclusion

This project demonstrates a complete Machine Learning workflow using Linear Regression to predict house prices.

Key learnings:

* Understanding datasets
* Data exploration and visualization
* Training and testing machine learning models
* Making predictions
* Evaluating model performance

Linear Regression is a simple and effective algorithm for regression problems and is a fundamental tool in Machine Learning.

---

## 10. Future Improvements

Possible improvements include:

* Using more advanced algorithms (Random Forest, Gradient Boosting)
* Feature engineering
* Hyperparameter tuning
* Model saving and deployment

---

## Author

Machine Learning Project – California House Price Prediction

