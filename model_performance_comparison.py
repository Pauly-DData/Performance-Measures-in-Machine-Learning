# Import necessary libraries
from pathlib import Path
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
import matplotlib.pylab as plt
import seaborn as sns
import numpy as np
from sklearn.metrics import confusion_matrix
import sklearn.metrics as pm
from sklearn.naive_bayes import MultinomialNB, GaussianNB

# Load the Diabetes dataset for KNN and Naive Bayes models
df = pd.read_csv(r'/diabetes.csv')

# Preparing data for modeling
X = df.drop('Outcome', axis=1).values # Dropping the 'Outcome' column for feature matrix
y = df['Outcome'].values # Using 'Outcome' column as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

# Display first few rows of the dataset
print(df.head())

# K-Nearest Neighbors Model
knn_model = KNeighborsClassifier(n_neighbors=7) # Initializing the KNN model with 7 neighbors
knn_model.fit(X_train, y_train) # Training the KNN model
y_pred = knn_model.predict(X_test) # Predicting using the KNN model

# Display predicted and actual values
print(y_pred) # Predicted values
print(y_test) # Actual values

# Computing and displaying the confusion matrix
tn, fp, fn, tp = pm.confusion_matrix(y_test, y_pred).ravel()
print("true negative:", tn)
print("false positive:", fp)
print("false negative:", fn)
print("true positive:", tp)

# Calculating performance measures for the KNN model
print("Performance measures over testing data set:")
print(" - precision is", pm.precision_score(y_test, y_pred))
print(" - recall is", pm.recall_score(y_test, y_pred))
print(" - f-measure is", pm.f1_score(y_test, y_pred))

# Evaluating performance on the training dataset
predictions = knn_model.predict(X_train)
print("Performance measures over training data set:")
print(" - precision is", pm.precision_score(y_train, predictions))
print(" - recall is", pm.recall_score(y_train, predictions))
print(" - f-measure is", pm.f1_score(y_train, predictions))

# Visualizing the confusion matrix using heatmap for the KNN model
cf_matrix = pm.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Blues', fmt='.2f')
ax.set_title('Confusion Matrix for KNN Model\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values');
ax.xaxis.set_ticklabels(['healthy (0)', 'diabetic (1)'])
ax.yaxis.set_ticklabels(['healthy (0)', 'diabetic (1)'])
plt.show()

# Naive Bayes Model
nb_model = MultinomialNB() # Initializing the Naive Bayes model
nb_model.fit(X_train, y_train) # Training the Naive Bayes model
y_pred = nb_model.predict(X_test) # Predicting using the Naive Bayes model

# Computing and displaying the confusion matrix for Naive Bayes model
tn, fp, fn, tp = pm.confusion_matrix(y_test, y_pred).ravel()
print("true negative:", tn)
print("false positive:", fp)
print("false negative:", fn)
print("true positive:", tp)

# Calculating performance measures for the Naive Bayes model
print("Performance measures over testing data set:")
print(" - precision is", pm.precision_score(y_test, y_pred))
print(" - recall is", pm.recall_score(y_test, y_pred))
print(" - f-measure is", pm.f1_score(y_test, y_pred))

# Visualizing the confusion matrix using heatmap for the Naive Bayes model
cf_matrix = pm.confusion_matrix(y_test, y_pred)
ax = sns.heatmap(cf_matrix, annot=True, cmap='Greens', fmt='.2f')
ax.set_title('Confusion Matrix for Naive Bayes Model\n\n');
ax.set_xlabel('\nPredicted Values')
ax.set_ylabel('Actual Values');
ax.xaxis.set_ticklabels(['healthy (0)', 'diabetic (1)'])
ax.yaxis.set_ticklabels(['healthy (0)', 'diabetic (1)'])
plt.show()

# Load the Iris dataset for the GaussianNB model
df = pd.read_csv(r'/Iris.csv')

# Preparing data for Gaussian Naive Bayes modeling
X = df.drop('species', axis=1).values # Dropping the 'species' column for feature matrix
y = df['species'].values # Using 'species' column as the target variable
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.40, random_state=101)

# Display first few rows of the Iris dataset
print(df.head())

# Initialize and train the Gaussian Naive Bayes model
GNB_model = GaussianNB()
GNB_model.fit(X_train, y_train)

# Predicting using the Gaussian Naive Bayes model
y_pred = GNB_model.predict(X_test)

# Example predictions with specific input values
# Prediction for input values [5.1, 2.5, 3.0, 1.1]
prediction1 = GNB_model.predict([[5.1, 2.5, 3.0, 1.1]])
print("Prediction for [5.1, 2.5, 3.0, 1.1]:", prediction1)

# Prediction for input values [6.5, 3.0, 5.5, 1.8]
prediction2 = GNB_model.predict([[6.5, 3.0, 5.5, 1.8]])
print("Prediction for [6.5, 3.0, 5.5, 1.8]:", prediction2)
