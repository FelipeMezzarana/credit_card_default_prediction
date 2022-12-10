# Credit Card Default Prediction

**General Objectives:**

This project aims to predict credit card customer default payments in the next month

Early detection of default payments would help financial institution on risk assessment and provide an objective decision making for the following:

+ Approval for future increase or decrease in credit facilities
+ Interest rates based on customer risk
+ Identify potential debt repayment schemes for targeted customers

**Learning Coverage:**

This project also aims to practice a series of techniques in relation to the development of a machine learning model, such as:

+ How to perform an EDA to understand the data (with good practices in data visualization and storytelling)
+ How to create features (feature engineering) based on the performed EDA and business knowledge
+ Which metrics and visualizations are important for evaluating a classification model, and how to use then. 
+ How to treat imbalanced dataset with fours different techniques (Under-Sampling, Over-Sampling, SMOTE, SMOTE + Under-Sampling)
+ How to build a Random Forest Classifier and a XGBoost classifier model 
+ How to evaluate a classification model using Stratified K-Folds cross-validator 
+ How to avoid data leakage (we carefully explore the correct ways to train and evaluate a model)
+ How to do hyperparameter tuning using Grid search
+ How to analyze model bias-variance trade-off with the learning curve

# Project Structure

To make the project easier to read and understand, I decided to divide it into three main files.

**Python notebook - eda_and_feature_engineering:**

+ Define the problem we want to solve 
+ Understand and vizualize the data (emphasis on best practices in data visualization and storitelling)
+ Look for potencial problems 
+ Create a few features

**Python notebook - default_predict:**

+ Define which metrics should be used considering the problem at hand 
+ Define functions to build a model (Random Forest and XGBoost) 
+ Define funtions to evaluate models with Stratified K-Folds cross-validator (avoiding data leakage)
+ Treat the imbalance data problem and choose the best technique
+ Feature selection and engineering (dealing with categorical features)
+ Model selection and hyperparameter tuning with grid search
+ Evaluate model performance with learning curve


**classification.py**

Using metrics for a classification model is something recurrent in any problem of this type. So, to make our work more scalable and our notebooks easier to read, I created a class Metrics and some functions in the file classification.py, that shall be imported in the notebook "default_predict".

The Metrics class receive two lists (list of predict default and list of actual default) and define the desired metrics as attributes. Methods:
+ ***show_metrics(self)*** -- Print metrics
+ ***show_charts(self, chart = 'all')*** -- Can plot metrics chart (may be: Confusion Matrix, ROC Curve or both) and display then. 
+ ***show_all(self)*** -- print metrics and display all charts. 
    
We also have two functions:

+ ***show_mean_metrics(metrics_obj_list, charts = 1)*** -- To print mean metrics, generate overlapped charts, and other useful infos for multiple Metrics objects. Please note that we will use Stratified K-Folds cross-validator to evaluate our models. Therefore, it will be interesting to have a function that can receive multiple metrics objects (one for each fold). 
+  ***mean_metrics(metrics_obj_list)*** -- Return a DataFrame with mean metrics for multiple tests (K-Folds)

