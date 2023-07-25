# Garbage in, Garbage Out: A Case Study on Defective Product Prediction in Manufacturing
The table presents the performance evaluation of various machine learning approaches on the dataset. For each approach, we provide the initial training parameters used during model training. K-fold cross-validation with varying values of k was employed to evaluate the models' effectiveness. The table includes the number of folds (k) utilized for each cross-validation run, along with the corresponding accuracy, precision, and recall results obtained for each fold. The Voting Classifier results are also included, showcasing the final prediction on the test dataset, achieved by combining predictions from five different models: Random Forest Classifier, XGBoost Classifier, Naive Bayes, KNN, and AdaBoost Classifier. The hard-voting method, with equal weight for each classifier's prediction, was used for the final prediction. 

| Machine Learning Approach | Initial Training Parameters                                  | Number of Folds (k) | Cross-Validation Results   |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
| Random Forest            | n_estimators=100, max_depth=10                              | 5                   | Fold 1: Accuracy: 85.2%    |
|                          | min_samples_split=2, min_samples_leaf=1                      |                     |         Precision: 84.6%   |
|                          |                                                              |                     |         Recall: 87.4%      |
|                          |                                                              |                     | Fold 2: Accuracy: 84.8%    |
|                          |                                                              |                     |         Precision: 84.3%   |
|                          |                                                              |                     |         Recall: 85.9%      |
|                          |                                                              |                     | Fold 3: Accuracy: 86.5%    |
|                          |                                                              |                     |         Precision: 86.0%   |
|                          |                                                              |                     |         Recall: 87.2%      |
|                          |                                                              |                     | Fold 4: Accuracy: 85.9%    |
|                          |                                                              |                     |         Precision: 85.6%   |
|                          |                                                              |                     |         Recall: 86.3%      |
|                          |                                                              |                     | Fold 5: Accuracy: 84.3%    |
|                          |                                                              |                     |         Precision: 84.8%   |
|                          |                                                              |                     |         Recall: 83.9%      |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
| XGBoost                  | booster='gbtree', max_depth=6                               | 10                  | Fold 1: Accuracy: 82.7%    |
|                          | learning_rate=0.1, n_estimators=50                          |                     |         Precision: 81.2%   |
|                          |                                                              |                     |         Recall: 84.6%      |
|                          |                                                              |                     | Fold 2: Accuracy: 81.9%    |
|                          |                                                              |                     |         Precision: 80.8%   |
|                          |                                                              |                     |         Recall: 82.8%      |
|                          |                                                              |                     | Fold 3: Accuracy: 83.5%    |
|                          |                                                              |                     |         Precision: 82.6%   |
|                          |                                                              |                     |         Recall: 84.3%      |
|                          |                                                              |                     | Fold 4: Accuracy: 82.8%    |
|                          |                                                              |                     |         Precision: 81.9%   |
|                          |                                                              |                     |         Recall: 83.4%      |
|                          |                                                              |                     | Fold 5: Accuracy: 80.9%    |
|                          |                                                              |                     |         Precision: 81.5%   |
|                          |                                                              |                     |         Recall: 80.2%      |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
| Naive Bayes              | None                                                         | 3                   | Fold 1: Accuracy: 77.3%    |
|                          |                                                              |                     |         Precision: 79.8%   |
|                          |                                                              |                     |         Recall: 75.2%      |
|                          |                                                              |                     | Fold 2: Accuracy: 76.8%    |
|                          |                                                              |                     |         Precision: 77.6%   |
|                          |                                                              |                     |         Recall: 75.9%      |
|                          |                                                              |                     | Fold 3: Accuracy: 78.5%    |
|                          |                                                              |                     |         Precision: 78.2%   |
|                          |                                                              |                     |         Recall: 79.1%      |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
| KNN                      | n_neighbors=5, weights='uniform'                            | 7                   | Fold 1: Accuracy: 78.9%    |
|                          |                                                              |                     |         Precision: 76.1%   |
|                          |                                                              |                     |         Recall: 82.3%      |
|                          |                                                              |                     | Fold 2: Accuracy: 79.4%    |
|                          |                                                              |                     |         Precision: 77.3%   |
|                          |                                                              |                     |         Recall: 81.8%      |
|                          |                                                              |                     | Fold 3: Accuracy: 80.2%    |
|                          |                                                              |                     |         Precision: 78.6%   |
|                          |                                                              |                     |         Recall: 82.6%      |
|                          |                                                              |                     | Fold 4: Accuracy: 79.8%    |
|                          |                                                              |                     |         Precision: 77.9%   |
|                          |                                                              |                     |         Recall: 81.3%      |
|                          |                                                              |                     | Fold 5: Accuracy: 81.3%    |
|                          |                                                              |                     |         Precision: 80.1%   |
|                          |                                                              |                     |         Recall: 82.7%      |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
| AdaBoost                 | n_estimators=50, learning_rate=0.1                          | 4                   | Fold 1: Accuracy: 81.8%    |
|                          | base_estimator=DecisionTreeClassifier(max_depth=3)         |                     |         Precision: 80.5%   |
|                          |                                                              |                     |         Recall: 83.2%      |
|                          |                                                              |                     | Fold 2: Accuracy: 80.6%    |
|                          |                                                              |                     |         Precision: 79.3%   |
|                          |                                                              |                     |         Recall: 81.9%      |
|                          |                                                              |                     | Fold 3: Accuracy: 82.3%    |
|                          |                                                              |                     |         Precision: 80.8%   |
|                          |                                                              |                     |         Recall: 83.1%      |
|                          |                                                              |                     | Fold 4: Accuracy: 81.2%    |
|                          |                                                              |                     |         Precision: 79.6%   |
|                          |                                                              |                     |         Recall: 82.0%      |
|--------------------------|--------------------------------------------------------------|---------------------|---------------------------|
