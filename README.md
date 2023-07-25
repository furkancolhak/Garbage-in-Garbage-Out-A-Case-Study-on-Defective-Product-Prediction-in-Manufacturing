# Garbage in, Garbage Out: A Case Study on Defective Product Prediction in Manufacturing
The table presents the performance evaluation of various machine learning approaches on the dataset. For each approach, we provide the initial training parameters used during model training. K-fold cross-validation with varying values of k was employed to evaluate the models' effectiveness. The table includes the number of folds (k) utilized for each cross-validation run, along with the corresponding accuracy, precision, and recall results obtained for each fold. The Voting Classifier results are also included, showcasing the final prediction on the test dataset, achieved by combining predictions from five different models: Random Forest Classifier, XGBoost Classifier, Naive Bayes, KNN, and AdaBoost Classifier. The hard-voting method, with equal weight for each classifier's prediction, was used for the final prediction. 
