Overview: Its a multiclass classification model that will predict the Iris flower species based on its sepal, and petal measurements.
This model classifies flowers into one of the three classes: Setosa, Versicolor, or Virginica.

# Pre-requisite libraries
    # import pandas as pd
        Used to load and manipulate the dataset as a dataframes.

    # import matplotlib.pyplot as plt
        Used to create basic visualizations like line or scatter plots.

    # import seaborn as sns
        Built on top of matplotlib, simplifies complex plots and adds better styling.

    from sklearn.datasets import load_iris
        Actual dataset that we use for training and prediction. load_iris() will loads the built-in Iris dataset from Scikit Learn.

    from sklearn.model_selection import train_test_split
        We will use train_test_split function to split the dataset for training and testing.

    from sklearn.linear_model import LogisticRegression
        We will apply LogisticRegression algorithm on the dataset to build model for classification that will make predictions.

    from sklearn.metrics import classification_report, accuracy_score, confusion_matrix
        We are importing few functions to evaluate our model performance or we can say to know how good our model is evaluating.

# iris = load_iris()        
    1. This will load the iris dataset from scikit learn library, which will be used to train and test our model.
    2. load_iris() returns a bunch object.
        Bunch object: It is a custom object that behaves like both a dictionary, and an object.
        You can access its elements using: 
            Dictionary-style: iris['data']
            Attribute-style: iris.data
        Both will return the same output.

# print(f"Fields in the iris dataset:\n{iris.keys()}")
    This prints all the fields(keys) available in the iris.dataset.
    Since, the iris object can also behaves like a dictionary(bunch object), that's why we can use keys() method on it. 
    Output: dict_keys(['data', 'target', 'frame', 'target_names', 'DESCR', 'feature_names', 'filename', 'data_module'])

# df = pd.Dataframe(data=iris.data, columns=iris.feature_names)
    Using the dataframe constructor from pandas, to create a DataFrame named 'df' from specific fields of the iris dataset.
    data=iris.data:
        This will load the numerical measurements of the input features of the iris flower(which is the length and width of sepal and petal of the iris flowers).
    columns= iris.feature_names
        This will set the column names in the DataFrame as the name of these input features. These features are: sepal length, sepal width, petal length, and petal width

# df['species'] = iris.target 
    This adds a new column to the dataframe 'df' named 'species'.
    The values in this 'species' column comes from iris.target(which is a key or field in the iris dataset), which contains numeric labels(0,1,2) representing the flower species.

# df['species_name'] =  df['species'].apply(lambda x: iris.target_names[x])
    Following things are happening here:
    df['species_name']: This adds a new column to the dataframe 'df' named 'species_name'.
    df['species'].apply(...): This will apply() a function defined as an argument to each value in 'species' column individually.
    df['species'].apply(lambda x: iris.target_names[x]):
        The value of 'x' at each iteration of lambda function will be the current value in 'species' column(which is 0/1/2).
        It will then return the value at index 'x' from iris.target_names numpy array.
            Values in iris.target_names are: ['setosa' 'versicolor' 'virginica']. So, if:
            iris.target_names[0], the value will be 'setosa'
            iris.target_names[1], the value will be 'versicolor'
            iris.target_names[2], the value will be 'virginica'

# print(df.head())
    This will display the first 5 rows of the dataframe 'df'.
    If you want to display few/more rows, you can pass a number of our choice as an argument, then those many rows will only be displayed at the console.
    For example: df.head(10) will display first 10 rows

# g = sns.pairplot(df.drop(columns=['species'],), hue='species_name')
    1. This will generate a grid of plots showing pairwise relationships between the numeric features of the dataframe "df".
    2. The pairplot function only considers numeric features (or features which have numeric values in them) for plotting.
    3. Even though species feature contain numbers 0,1,2, but it actually represents iris flower species as setosa, versicolor, virginica respectively.
       That is the reason we will drop this feature, so that pairplot won't consider it.
    4. The hue parameter determines how the data points are color-coded in the plots.
       Setting hue='species_name' uses the actual species_name(setosa, versicolor, virginica) to differentiate the classes by color.
    5. For example, in a plot of sepal_length vs petal_width, the color at different points will show which species it belongs to, helping us visually understand
       how well the feature separates different flower classes.

# g.fig.suptitle("Pairplot of Iris Dataset", y = 1.0)
    This will add a title "Pairplot of Iris Dataset" to the top of the entire pairplot figure. 
    The y parameter controls the vertical position of the title relative to the figure.
        y=1.0 places the title directly at the top edge of the plot area.
        y>1.0 pushes the title slightly above the top edge.
        y<1.0 pushes the title slightly below the top edge or even into the plot area.
    
        Why g.fig.suptitle() to add a title,and not g.suptitle(). Doing so will throw an Attribute Error.
        Incase you are wondering about this, below is the reasoning:
            1. sns.pairplot() returns a seaborn.PairGrid object, which is stored in variable g.
            2. PairGrid contains a Matplotlib Figure inside it, which can be accessed via g.fig.
            3. suptitle() is a method of that Figure, so we must call it through g.fig.
            4. Calling g.suptitle() will throw an AttributeError as this method is not available in the PairGrid itself.

            If we want to add a title for a specific sub plot then we can do so by: g.axes[0,0].set_title("Title for subplot 0,0"),
            where [0, 0] indicates the row and column values of subplot in the Grid.

# plt.show()
    This command will display the final plot in a window. It is used when using Matplotlib to ensure visualizations render properly, especially in a script based environment
    like vscode or when directly running the python script file.

# X = iris.data
    This will store the feature data into the variable X(aka input variable).
    These includes values for: sepal length, sepal width, petal length, petal width for each sample.

# y = iris.target 
    This will store the target labels into the variable y(aka output variables).
    Each value corresponds to a species class: 0 - setosa, 1 - versicolor, 2 - virginica

# X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
    1. The train_test_split() will split the dataset among training and testing subsets.
    2. test_size = 0.2 indicates that, reserve 20% for testing, and the remaining 80% dataset for training the model.
    3. random_state=42 parameter will set the start point of randomization for the model. This  ensures reproducibility - it sets a fixed seed for random generator 
    so the split remains consistent everytime we run the code.

# model = LogisticRegression(max_iter=200)
    1. This will create an instance of LogisticRegression model class, and set the max_iterations value to 200. 
    2. The reference of LogisticRegression class object, will be stored in model variable.
    2. max_iter indicates the maximum number of optimization steps(iterations) the model can perform to adjust its input parameters(w and b) to minimize the loss function.
    The goal of these iterations is to find the best value of input parameters(w and b) such that the loss or error between the predicted label by the model and actual target 
    label is minimum.
    3. Its not mandatory to perform 200 iterations to compute the best value of input parameters, if we converge(the loss stops to reduce significantly with the change
    in the value of w and b) early, we can stop before 200 iterations as well. 

# model.fit(X_train, y_train)
    1. This will train the LogisticRegression model on the training dataset X_train(features) and y_train(target labels).
    2. We will also optimize our model's input parameters(w and b) by comparing its predicted label against the actual target label from the training data y_train.
    3. As stated above, at max we can perform 200 iterations to converge(beyond which further optimization of input parameters do not cause any significant reduction in the
    loss function) our model. If we converge before 200 iterations we can stop early as well.
    4. To optimize the value of input parameters(w and b) the model may use gradient descent algorithm or one of its varient.

# y_pred = model.predict(X_test)
    We are asking our trained model to make predictions now.
    The unseen test data, X_test, is given as input to the model for making predictions.
    The labels predicted by the model is stored in y_pred variable.

# print(confusion_matrix(y_test, y_pred))
    1. This statement will generate a confusion matrix against our model's predicted label(y_pred) vs actual labels(y_test), and display it at the console.
    2. Since the Iris dataset is a multiclass classification problem with 3 classes(setosa, versicolor, virginica), the confusion matrix will be a 3*3 matrix.
    where, row represents the actual labels, and column represents the predicted labels.
    Note: The sum of all elements in the confusion matrix is equal to the total number of input samples in the test dataset. 

        # Confusion matrix
            1. It is a function from scikit learn that gives you  a summary of how well your model is predicting the target labels against the actual labels, which inputs features
            it is able to predict correctly, and where it made mistakes. 
            2. It is particularly useful for understanding which classes our model is getting confused.

            The confusion matrix for a binary clasification problem will look as below:
                                    Predicted: No              Predicted: Yes            
                Actual: No          True Negative(TN)           False Positive(FP)
                Actual: Yes         False Negative(FN)          True Positive(TP)
                Binary means it has 2 possible output classes to classify from.
                The row indicates the actual labels, whereas the columns indicates the predicted labels.

            For example: 
                y_true = [0, 1, 2, 2, 0, 1]
                y_pred = [0, 0, 2, 2, 0, 2]
                cm = confusion_matrix(y_true, y_pred)
                print(cm)

                The output will be a square matrix of 3*3, as it is a multiclass classification problem which has 3 classes 0, 1,2.
                        0       1       2
                0       2       0       0
                1       1       0       1
                2       0       0       2

                The diagonal of the confusion matrix indicates, how many labels are predicted correctly, whereas the off-diagonal values indicates how many values are
                incorrectly predicted.For example the value at position [0][0] indicates that 2 samples of class 0 are predicted correctly, whereas the element at index
                [1][0] indicates that 1 sample of class 1 is misclassified as class 0 sample by the model.

# print(classification_report(y_test, y_pred, target_names=iris.target_names))
    This will generate a detailed summary about the performance of our classification model, and display it at the console.
    y_test => Actual label from the test dataset.
    y_pred => Predicted label by the classification model.
    target_names=iris.target_names => This is used to display names matching the labels.

    It shows the following metrics for each class:
    1. Precision: Out of predicted positives, how many were correct? 
        Precision = TP / TP + FP
    2. Recall: Out of actual positives, how many were correctly predicted?
        Recall = TP / TP + TN
    3. F1-score: This is the harmonic mean of recall and precision.
        F1-score = 2 * ((Precision * Recall) / (Precision + Recall))
    4. Support: Shows how many samples were present for each class.

    We also have other fields as well:
    1. Accuracy : Total correct predictions / Total number of samples.
        Its a global metric, hence, its is not calculated for each class but for the overall classification model. That's why it appears only once, on the f1-score column as 
        an overall score.
    2. Weighted avg : (Score for each class * Support value of that class)/ Total number of samples. 
        This is calculated for each field(Precision, Recall, F1-score) individually.
    3. Macro avg : Average of each class scores, treating all classes equally.
        Suppose we are calculating the macro average for Precision field(column), then the formula for that will be:
        macro avg of Precision = (Value of Precision for Class 1 + Value of Precision for Class 2 + --- + Value of Precision for Class n)/support value for any class(as it treats 
        all the classes equally)
        In our case, 
        macro avg of Precision = (Value of Precision for Class Setosa + Value of Precision for Versicolor + Value of Precision for Virginica ) / support value of Setosa/Versicolor/
        Virginica.

# print(f"Accuracy Score: {accuracy_score(y_test, y_pred):.2f}")
    1. The accuracy_score() will compare the predicted labels(y_pred) against actual labels(y_test), and returns the fraction of correct predictions.
        Formula for accuracy = Number of correct predictions / Total number of predictions.
    2. Lets say our model was provided 10 input samples to make predictions, and 6 out of 10 labels were predicted correctly. Then, the accuracy score of our model will be 0.6.
    3. .2f ensure that 2 values will be printed after the decimal places.
