import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

from sklearn.datasets import load_iris
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC


from sklearn.metrics import classification_report, accuracy_score, confusion_matrix

import numpy as np

def load_data():
    """Loads the Iris data"""
    iris = load_iris()
    X = iris.data
    y = iris.target
    feature_names = iris.feature_names
    target_names = iris.target_names
    return X, y, feature_names, target_names


def convert_to_dataframe(input_data, feature_names, actual_target_value,target_names):
    """Converting the iris data to dataframes,and modifying it"""
    df = pd.DataFrame(data = input_data, columns = feature_names)
    df['species'] = actual_target_value
    df['species_name'] = df['species'].apply(lambda x: target_names[x])
    return df

def visualize_dataframe(dataframe,):
    """Visualize the dataframe, and saves the generated plot in the current directory."""
    g = sns.pairplot(dataframe.drop(columns='species'), hue='species_name')
    g.fig.suptitle("Pairplot of Iris Dataset", y=1.0)
    # plt.show() Commenting this line because it was haulting the program in between whenever this line was hit, instead, I am saving the generated plot in a file.
    print("Generating a dataframe plot, and saving it.")
    plt.savefig(f'dataframe_plot.png')
    plt.close()

def split_data_for_training_testing(X, y, test_size=0.2, random_state=42):
    """Split the entire dataset into 2 parts for training and testing the model."""
    return train_test_split(X, y, test_size=test_size, random_state=random_state)


def preprocess(X):
    """Pre processing the data to remove any duplicates, or any kind of data which is confusing for the model"""
    scaler = StandardScaler()
    x_scaled = scaler.fit_transform(X)
    return x_scaled

# Create and train a model
def train_model(X_train, y_train, optimization_algo = "logistic"):
    """Create a model, and also train the model by optimizing its parameters on the optimization algo. received as input parameter."""
    if optimization_algo == "logistic":
        used_optimization_algo = "LogisticRegression"
        model = LogisticRegression(max_iter=200)
    elif optimization_algo == "random_forest":
        used_optimization_algo = "RandomForestClassifier"
        model = RandomForestClassifier()
    elif optimization_algo == "svc":
        used_optimization_algo = "SVC"        
        model = SVC()
    else:
        used_optimization_algo = "Unknown"
        raise ValueError(f"Unsupported optimization algorithm: {optimization_algo}")
    model.fit(X_train, y_train)
    return model, used_optimization_algo


# Make predictions 
def make_prediction(X, model):
    """Receiving input from the user(X), and using the trained model to make predictions."""
    return model.predict(X)

# Evaluate the model
def evaluate_model(y_test, y_pred, target_names, optimization_algo_used):
    """
        Evaluating the performance of the model by:     
            1. checking its accuracy.
            2. Generating a classification report, which will show detailed analysis about what's it precision rate, accuracy, weighted average etc.
            3. Building a confusion matrix, which shows how our model is predicting across each class, and for which class it is getting confused.
            4. Saving the generated confusion matrix plot as a file in current directory.
    """
    model_accuracy = accuracy_score(y_test, y_pred)
    print(f"Model accuracy using {optimization_algo_used} algorithm is: {model_accuracy:.2f}")
    
    cfr = classification_report(y_true=y_test, y_pred=y_pred, target_names=target_names)
    print(f"\nClassification report using {optimization_algo_used} algotithm for optimization is -\n{cfr}")
    
    cm = confusion_matrix(y_true=y_test, y_pred=y_pred)
    print(f"\nConfusion Matrix using {optimization_algo_used} algorithm is :\n{cm}")
    print(f"\nGenerating a plot from confusion matrix data, and saving it.")
    plt.figure(figsize=(6,5))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=target_names, yticklabels=target_names)
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    # plt.show()  Commenting this line because it was haulting the program in between whenever this line was hit, instead, I am saving the generated plot in a file.
    plt.savefig(f"{optimization_algo_used}_confusion_matrix_plot.png")
    plt.close()
    
def duplicated_input_data(input_data, samples, features=0):
    """
        1. Increasing the size of input data by duplicating the input data into it.
        2. samples: Number of times to repeat the samples in input data for building the new input data.
        3. features: Number of times to repeat the features in input data for building the new input data. Default value is 0, because if its 1-D array, then we dont need have this 
        dimension in the input data.
        4. Returning the updated input data.
    """
    if isinstance(input_data, np.ndarray):
        if input_data.ndim == 1:
            return np.tile(input_data, reps=(samples,))
        elif input_data.ndim == 2:
            return np.tile(input_data, reps=(samples, features))
        else:
            print("Numpy arrays beyond 2_d is received. Currently, its not supported")
    else:
        print("Received user input is not a numpy array.")

def main():
    X, y, feature_names, target_names = load_data()
    X = duplicated_input_data(input_data=X, samples=100, features=1)
    y = duplicated_input_data(input_data=y, samples=100)
    X = preprocess(X=X)
    X_train, X_test, y_train, y_test =  split_data_for_training_testing(X, y, test_size=0.2, random_state=42)
    model, optimization_algo_used = train_model(X_train, y_train, optimization_algo = "random_forest")
    y_pred = make_prediction(X=X_test, model=model)
    evaluate_model(y_test=y_test, y_pred=y_pred, target_names=target_names, optimization_algo_used=optimization_algo_used)
    df = convert_to_dataframe(input_data=X, feature_names=feature_names, actual_target_value=y, target_names=target_names)
    # print(df.head)
    visualize_dataframe(dataframe=df)


if __name__=="__main__":
    main()