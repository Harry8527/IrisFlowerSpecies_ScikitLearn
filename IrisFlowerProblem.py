import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

# Load the Iris dataset
iris = load_iris()
print(f"Fields in the iris dataset:\n{iris.keys()}")
# print(iris.data)
print(iris.feature_names)
print(iris.target_names)
print(iris.target)
# Convert it to a pandas DataFrame for easier handling
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)
print("\n\n")
# print(f"Values in target column of iris: {iris.target}")
df['species'] = iris.target
df['species_name'] =  df['species'].apply(lambda x: iris.target_names[x])
# print("\n\n")
print(df.head(10))

# Visualize the data
g = sns.pairplot(df.drop(columns=['species'],), hue='species_name')
print(type(g))
g.fig.suptitle("Pairplot of Iris Dataset", y = 1.0)
plt.show()

# Features and labels
X = iris.data
y = iris.target 

# Split the dataset into training and testing sets(80% train, 20% test)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
# print(f"Training data:\n{X_train}")
# print(f"Test data:\n {X_test}")


# Create and train a model
model = LogisticRegression(max_iter=200)
model.fit(X_train, y_train)

# Make predictions
y_pred = model.predict(X_test)

# Evaluate the model
print("\nConfusion Matrix:")
print(confusion_matrix(y_test, y_pred))
      
print("\nClassification Report:")
print(classification_report(y_test, y_pred, target_names=iris.target_names))

print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")