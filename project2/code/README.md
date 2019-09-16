# CIFAR-10 Classification

Code implemented for 4 types of classifiers: Linear SVM, RBF Kernel SVM, Logistic Regression, Decision Tree. Data representations: raw data, PCA, LDA.

### Requirements

* fire
* sklearn

### Scripts

##### Decision Tree
To see variation of accuracy of Decision Tree Classifier with max_depth.
* decision-tree-raw-data.py
* decision-tree-pca.py
* decision-tree-lda.py

##### RBF Kernel SVM
To see variation of accuracy of RBF Kernel SVM with C and gamma. Uses grid search to find optimal hyperparameters.
* rbf-kernel-svm-grid-search.py

##### Logistic Regression
To see variation of accuracy of Logistic Regression Classifier with solver. Calculates accuracy using "newton-cg", "lbfgs", "liblinear", "sag" & "saga" solvers.
* logistic-regression-solver.py

##### Linear SVM
To see variation of accuracy of Linear SVM with C.
* linear-svm-C.py

### run.py
* main code with all the classifier implementations
* Run using `python3 run.py --num_batches <num_batches> --data_representation <data_representation> --num_components <num_components> --classifier_type <classifier_type> --C <C> --solver <solver> --log_gamma <log_gamma> --log_C <log_C> --max_depth <max_depth>`
* All flags are optional. Default values are given inside the main function.
