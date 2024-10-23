# MeXE402_Midterm_-Evangelista-Malabanan-
This is repository for Midterm in Elective 2
## **Introduction (Overview of Linear Regression)**

&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Linear regression is one of the most fundamental algorithms in supervised machine learning. It is a method used to model the relationship between a dependent variable (target) and one or more independent variables (features). In simple linear regression, we aim to predict a continuous output variable based on one input feature by fitting a line to the data that minimizes the difference between the predicted and actual values. When multiple features are used, the model becomes a multiple linear regression. Linear regression models are widely applied in various fields, including economics, biology, and real estate, to understand and predict trends. In this project, linear regression will be applied to predict the median value of homes in different Boston suburbs based on several socio-economic and environmental factors.
<br>
## **Introduction (Overview of Linear Regression for the Given Dataset)**
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;Linear regression is a fundamental statistical technique used to model the relationship between a dependent variable (target) and one or more independent variables (features). In this project, we will apply multiple linear regression to explore how different socio-economic and environmental factors influence the housing market in Boston suburbs, using a dataset that records various attributes of the towns in the Boston area.
<br>
<br>
&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;&nbsp;The goal of this linear regression analysis is to predict the median value of owner-occupied homes (the target variable, MEDV) using a variety of features such as crime rate (CRIM), the average number of rooms per dwelling (RM), proximity to highways (RAD), and others. By building a linear regression model, we aim to quantify the strength of the relationships between these features and housing prices.
Through this process, we will be able to answer key questions, such as:
<br>
<br>
How do environmental factors like air quality (NOX) or proximity to the Charles River (CHAS) impact home values?
<br>
Do higher property taxes (TAX) and pupil-teacher ratios (PTRATIO) lead to lower housing prices?
<br>
Can socio-economic factors like the crime rate (CRIM) or percentage of lower-status residents (LSTAT) be used to explain fluctuations in real estate prices?
<br>
The linear regression model will help provide insights into the Boston housing market by offering a data-driven approach to predicting property values, which could be valuable for urban planners, real estate investors, or economists studying the area.
<be>

## **Dataset Description**
The dataset contains information about various factors that may influence the value of homes in Boston suburbs, derived from the Boston Standard Metropolitan Statistical Area (SMSA) in 1970. It includes 505 entries, each representing a different town or suburb. The dataset consists of the following features:<br>
<br>
**CRIM:** Per capita crime rate by town
<br>
**ZN:** Proportion of residential land zoned for large lots
<br>
**INDUS:** Proportion of non-retail business acres per town
<br>
**CHAS:** Whether the town bounds the Charles River (1 for yes, 0 for no)
<br>
**NOX:** Nitric oxides concentration
<br>
**RM:** Average number of rooms per dwelling
<br>
**AGE:** Proportion of owner-occupied units built before 1940
<br>
**DIS:** Weighted distances to employment centers
<br>
**RAD:** Index of accessibility to highways
<br>
**TAX:** Property tax rate per $10,000
<br>
**PTRATIO:** Pupil-teacher ratio
<br>
**B:** A measure of racial diversity
<br>
**LSTAT:** Percentage of lower-status residents
<br>
**MEDV:** Median value of homes, which is the target variable in this analysis.
The goal is to use these features to predict MEDV, providing insight into the factors that most influence housing prices.

## Project Objectives
The main objective of this project is to use linear regression to predict the median value of homes in Boston suburbs. Through the analysis, we aim to:
Identify which features have the most significant impact on housing prices.
Build a regression model that can predict housing prices based on various factors, such as crime rate, number of rooms, and proximity to employment centers.
Evaluate the performance of the model using statistical metrics such as the coefficient of determination (R-squared) and root mean square error (RMSE).
Analyze the model’s residuals to assess its accuracy and identify potential areas for improvement.
1. Model's Ability to Predict (R² Value = 0.6688):
The R² value of 0.6688 indicates that approximately 66.88% of the variance in the median home values (the target variable) can be explained by the independent variables in your model. This suggests that your model has a moderate to strong predictive power. While it is capturing a good amount of variance in the data, there is still room for improvement, as 33.12% of the variance is not accounted for.
In real-world datasets, especially those involving socio-economic or environmental factors like this one, an R² value between 0.6 and 0.7 is fairly common and considered good, as human behavior and real estate prices are influenced by many factors that can be hard to capture entirely.
2. Comparison of Actual vs. Predicted Values:
Looking at the actual and predicted values:
Actual Values
Predicted Values
23.6
28.99
32.4
36.03
13.6
14.82
22.8
25.03
16.1
18.77
20.0
23.25
17.8
17.66
14.0
14.34
19.6
23.01
16.8
20.63

While the predictions are generally close to the actual values, there are some deviations. For instance, in the first case, the actual value is 23.6, but the model predicts 28.99, an overestimation. In the third example, the prediction (14.82) is close to the actual value (13.6), indicating better performance for this case.
These discrepancies can be attributed to several factors:
Model's assumptions: Linear regression assumes a linear relationship between the target and the predictors, which may not fully capture complex interactions.
Data distribution: Outliers or skewed distributions in certain features can cause the model to be less accurate in some predictions.
Unexplained variance: The 33.12% of variance that the model doesn’t capture could be due to missing variables or non-linear relationships.

The cross-validation Mean Squared Error (MSE) of 24.2911 provides insight into how well your model is expected to perform on unseen data. Let’s break down what this means:
Understanding Mean Squared Error (MSE):
The MSE measures the average of the squares of the errors (the differences between actual and predicted values). 

MSE of 24.2911 means that, on average, the squared difference between the actual and predicted home values is around 24.2911 units squared. Since your target variable (home value) is in thousands of dollars, the MSE of 24.2911 means that, on average, the square of the error is 24,291

Cross-validation splits the data into multiple folds (subsets), trains the model on a subset of the data, and tests it on the remaining part. This process is repeated to reduce the bias from any particular split.

A cross-validated MSE is a more reliable measure of model performance since it estimates how the model generalizes to unseen data. Your MSE value of 24.2911 suggests that the predictions made by the model are off by a considerable margin when squared.

While MSE is in squared units, you can compute the Root Mean Squared Error (RMSE) to interpret it in the same units as the target variable

This means that the typical error in predicting the median home value is approximately $4,928 (in thousands of dollars).

The lower the MSE/RMSE, the better your model is at making predictions.

The current RMSE value (about 4.928) suggests that your model is, on average, around $4,928 off from the actual home prices. Depending on the scale of the data, this might be acceptable or might suggest further refinement is needed.

In a linear regression model, each feature is assigned a coefficient that represents the impact that feature has on the target variable (in this case, the median value of homes). Let’s break down the significance of these coefficients and the overall predictive power of your model.
1. Significance of Coefficients:
The coefficients in a linear regression model can tell you how much the target variable changes for a unit change in the corresponding feature, assuming all other features are held constant. Here’s how to interpret them:
Positive Coefficient: A positive value means that an increase in that feature is associated with an increase in the predicted home value.
For example, if the coefficient of RM (average number of rooms) is positive, then homes with more rooms tend to have higher values.
Negative Coefficient: A negative value indicates that an increase in the feature is associated with a decrease in the predicted home value.
For example, if the coefficient of LSTAT (percentage of lower-status residents) is negative, then as the percentage of lower-status residents increases, the home values tend to decrease.
Magnitude of Coefficients: The size of the coefficient reflects the strength of the impact on the target variable.
A larger coefficient (in absolute value) means the feature has a stronger impact on the home prices.
For example, if RM has a coefficient of 5 and TAX has a coefficient of -0.1, an additional room increases home value by $5,000, while a higher tax rate slightly decreases it by $100.
Significance of Features (p-values): The p-value (if available) indicates whether a coefficient is statistically significant.
A low p-value (typically < 0.05) means that the corresponding feature has a significant impact on the target variable.
A high p-value suggests that the feature might not be contributing meaningfully to the model and could potentially be removed.

If we assume the following coefficients from your model:
RM (rooms): +4.5 → For each additional room, the predicted home value increases by $4,500.
CRIM (crime rate): -1.2 → For each unit increase in crime rate, the predicted home value decreases by $1,200.
LSTAT (lower-status population): -2.0 → For every 1% increase in lower-status residents, the home value decreases by $2,000.
This tells you that homes with more rooms tend to have higher values, while neighborhoods with higher crime rates or a higher percentage of lower-status residents tend to have lower home values.

2. Model’s Predictive Power:
The predictive power of your model is primarily evaluated through the following metrics:
R² (R-squared): In your case, the R² value is 0.6688, meaning the model explains 66.88% of the variance in the target variable (home prices). This suggests that the model captures a good portion of the variation in home prices based on the given features, but there is still 33.12% of the variance that the model does not account for. This could be due to missing features, non-linear relationships, or other factors not captured in this model.
Cross-Validation MSE (Mean Squared Error): The cross-validation MSE of 24.2911 gives an idea of the model’s performance on unseen data. This value indicates how far off, on average, the squared errors are between the actual and predicted home values. The corresponding RMSE (around 4.93) suggests that the typical prediction error is about $4,930 when predicting home prices.

METHODOLOGY:
1. Importing Necessary Libraries
The first step is to import the libraries that will help you handle the data, visualize the results, and perform the machine learning operations.
pandas (pd): Handles data in table format.
numpy (np): Used for numerical operations.
matplotlib & seaborn: Used to create graphs.
scikit-learn: The machine learning library that provides functions to split data, scale it, create a model, and evaluate performance.

2. Loading and Understanding the Dataset
You load the Boston housing dataset and check for any missing or problematic data.
df.info(): Shows details about each column (like data type and number of entries).
df.isna().sum(): Ensures there are no missing values in the dataset.
3. Separating Features (X) and Target (y)
You define the columns used to predict house prices and the actual house price column.
X: Features like crime rate, number of rooms, etc.
y: The price of the house.
4. Splitting the Data for Training and Testing
You split the dataset into training data (to build the model) and testing data (to evaluate the model).
80% Training data: Used to train the model.
20% Testing data: Used to test the model's accuracy.
5. Standardizing (Scaling) the Data
You scale the features so that they are all on a similar scale. This helps the model perform better.
StandardScaler: Makes sure all features have the same scale, so the model isn't biased toward features with larger numbers.
6. Training the Model
You train the linear regression model on the training data.
Linear Regression: A model that tries to draw a straight line through the data to predict house prices.
7. Making Predictions and Evaluating the Model
After training, you test the model's performance on unseen test data and evaluate how well it predicts house prices.
Mean Squared Error (MSE): Tells you how far, on average, the predicted prices are from the actual prices.
R² Score: Tells you how well the model explains the variation in house prices (1 is perfect).
8. Comparing Actual vs. Predicted Values
You compare the actual prices to the predicted prices in a table for a quick look at how close the model’s predictions are.
9. Visualizing Predictions (Scatter Plot)
You create a scatter plot to visualize the accuracy of the model's predictions.
Scatter Plot: Shows how well the predicted prices align with the actual prices.
Red Line: Represents a perfect prediction.

10. Visualizing Errors (Residual Plot)
You create a residual plot to visualize how far the predictions are from the actual values.
Residual Plot: Shows the errors. If points are randomly scattered around the red line, it indicates good model performance.

11. Making Predictions on New Data
Finally, you use the trained model to predict the price of a new house based on its features.


REFERENCES
Pedregosa, F., Varoquaux, G., Gramfort, A., Michel, V., Thirion, B., Grisel, O., ... & Duchesnay, É. (2011). Scikit-learn: Machine learning in Python. Journal of Machine Learning Research, 12, 2825-2830. Available from https://scikit-learn.org/stable/index.html 

McKinney, W. (2010). Data structures for statistical computing in Python. In Proceedings of the 9th Python in Science Conference (pp. 51-56). Available from https://pandas.pydata.org/pandas-docs/stable/ 

Hunter, J. D. (2007). Matplotlib: A 2D graphics environment. Computing in Science & Engineering, 9(3), 90-95. doi:10.1109/MCSE.2007.55 Available from https://matplotlib.org/stable/users/index.html 

Waskom, M. (2021). Seaborn: Statistical data visualization. Journal of Open Source Software, 6(60), 3021. Available from https://seaborn.pydata.org/ 

UCI Machine Learning Repository (n.d.). Boston Housing Data. Retrieved from https://archive.ics.uci.edu/ml/datasets/Housing 





