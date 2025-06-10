1. Dimension of input data after duplicating the data: (15000, 4), which means the data has 15000 samples and each sample has 4 features in it.

2. Accuracy of the model across after training with different optimization algorithms is the following:

With LogisticRegression: 0.98

With RandomForestClassifier: 1.00

With SVC: 0.99

3. The classification report generated using different optimization algorithm is the following:


    Using SVC algotithm -
                precision    recall  f1-score   support

        setosa       1.00      1.00      1.00       983
    versicolor       0.98      0.98      0.98      1002
    virginica       0.98       0.98      0.98      1015

    accuracy                             0.99      3000
    macro avg       0.99      0.99       0.99      3000
    weighted avg       0.99   0.99       0.99      3000


    Using LogisticRegression algorithm -
               precision    recall  f1-score   support

      setosa       1.00      1.00      1.00       983
   versicolor      0.98      0.95      0.97      1002
   virginica       0.95      0.98      0.97      1015

   accuracy                            0.98      3000
   macro avg       0.98      0.98      0.98      3000
   weighted avg    0.98      0.98      0.98      3000

    Using Random Forest Classifier algorithm -

              precision    recall  f1-score   support

      setosa       1.00      1.00      1.00       983
  versicolor       1.00      1.00      1.00      1002
   virginica       1.00      1.00      1.00      1015

    accuracy                           1.00      3000
   macro avg       1.00      1.00      1.00      3000
   weighted avg    1.00      1.00      1.00      3000



4. Confusion Matrix with different optimization algorithms is the following:

    Using SVC algorithm :
        [[983   0   0]  
        [  0 982  20]
        [  0  18 997]]

    ![SVC algorithm plot](SVC_confusion_matrix_plot-1.png)

    Using LogisticRegression algorithm:
        [[983   0   0]
        [  0 953  49]
        [  0  18 997]]
    
    ![Logistic Regression plot](LogisticRegression_confusion_matrix_plot.png)

    Using RandomForestClassifier algorithm:
        [[ 983    0    0]
        [   0 1002    0]
        [   0    0 1015]]
    
    ![Random Forest Classifier plot](RandomForestClassifier_confusion_matrix_plot.png)


Conclusion: From the above metrics, it seems that Random Forest Classifier algorithm works best for classification with large data.