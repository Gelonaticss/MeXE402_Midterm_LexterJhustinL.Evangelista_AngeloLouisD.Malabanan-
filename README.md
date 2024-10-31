# Midterm Project for Elective 2

## **Linear Regression**

### **What is Linear Regression**

- **Linear regression** is a fundamental algorithm in *supervised machine learning*.
- Used to model relationships between a **dependent variable** (*target*) and one or more **independent variables** (*features*).
- In *simple linear regression*, a single feature is used to predict a continuous output variable.
- *Multiple linear regression* uses several features to predict a continuous output.
- Widely applied across fields like **economics**, **biology**, and **real estate** to analyze and predict trends.
- In this project, linear regression will predict the **median value of homes** in various **Boston suburbs** based on socio-economic and environmental factors.

### **Overview of Linear Regression for the Boston Dataset**

- **Goal**: Predict the **median value of owner-occupied homes** (*target variable: MEDV*) based on factors such as:
  - **CRIM** - *Crime rate per capita*
  - **RM** - *Average number of rooms per dwelling*
  - **RAD** - *Proximity to highways*
- Key questions addressed include:
  - How environmental factors like **air quality** (*NOX*) or **proximity to Charles River** (*CHAS*) impact home values.
  - Whether **property taxes** (*TAX*) and **pupil-teacher ratios** (*PTRATIO*) correlate with lower housing prices.
  - If socio-economic factors, such as **crime rate** (*CRIM*) or **percentage of lower-status residents** (*LSTAT*), explain fluctuations in real estate prices.

## **Dataset Description**

- **Boston Housing Dataset** contains data from the 1970 Boston Standard Metropolitan Statistical Area (*SMSA*).
- **Entries**: 505 towns/suburbs.
- **Features**:
  - **CRIM**: *Per capita crime rate by town*
  - **ZN**: *Proportion of residential land for large lots*
  - **INDUS**: *Non-retail business acres per town*
  - **CHAS**: *Town proximity to Charles River (1 = Yes, 0 = No)*
  - **NOX**: *Nitric oxide concentration*
  - **RM**: *Average number of rooms per dwelling*
  - **AGE**: *Owner-occupied units built before 1940*
  - **DIS**: *Distance to employment centers*
  - **RAD**: *Highway accessibility index*
  - **TAX**: *Property tax rate per $10,000*
  - **PTRATIO**: *Pupil-teacher ratio*
  - **B**: *Racial diversity measure*
  - **LSTAT**: *Percentage of lower-status residents*
  - **MEDV**: *Median home value* - the target variable for prediction.
 
## Why `MEDV` Should be the Dependent Variable
- `MEDV` represents the median home price, which is typically the main target for prediction in real estate datasets.
- Predicting `MEDV` can provide insights into housing market trends, affordability, and property valuations in Boston’s suburbs.
- Most features in this dataset (e.g., crime rate, number of rooms, accessibility to highways) influence home values, making `MEDV` an appropriate target for regression analysis.

## Data Type of `MEDV`
- `MEDV` is **continuous**, as it represents home prices, which can take a range of numerical values, not limited to categories or fixed intervals.

## Appropriateness for Linear Regression
- Since `MEDV` is a continuous variable, it is suitable for linear regression, which predicts continuous outcomes.
- Linear regression assumes a linear relationship between the target and predictor variables. This can help model how various factors like room count or crime rate contribute to home price fluctuations.

## Initial Observations of the Dataset
- **Potential Non-Linear Relationships:** Some features (e.g., crime rate) may have a non-linear relationship with `MEDV`, which might affect linear regression's performance.
- **Outliers in `MEDV`:** The dataset may contain outliers in home prices, which could skew predictions and reduce model accuracy.
- **Feature Multicollinearity:** There might be correlations among features (e.g., `NOX` and `INDUS`), which could introduce multicollinearity issues.
- **Skewed Distributions:** Certain variables may have skewed distributions, potentially influencing regression results and suggesting the need for feature transformation or regularization.


## **Project Objectives**

1. **Feature Impact on Housing Prices**:
   - Identify which **features significantly impact** housing prices.
2. **Model Building**:
   - Build a regression model to predict **housing prices** using socio-economic factors.
3. **Model Evaluation**:
   - Evaluate using **R-squared** and **RMSE** to assess model performance.
4. **Residual Analysis**:
   - Analyze residuals to evaluate accuracy and identify improvement areas.


## Methodology

1. **Importing Libraries and Loading Data**
   - Imported necessary libraries, including `pandas`, `numpy`, `seaborn`, and `matplotlib` for data processing, visualization, and inline plotting.
   - Loaded the dataset `clearboston.csv` into a DataFrame.

2. **Initial Data Exploration**
   - Displayed the first five rows to understand the data structure.
   - Checked for missing values in the dataset to ensure data completeness.
   - Used `.info()` and `.describe()` to explore data types and summary statistics.

3. **Identifying Key Observations in Data**
   - Noted interesting patterns:
     - `ZN` and `CHAS` columns showed minimal variability, indicating potential limited predictive value.
     - `MEDV` values above 50 were capped, highlighting the need to handle this censored data.
   - Plotted boxplots to visualize outliers and noted columns like `CRIM`, `ZN`, `RM`, and `B` had high outlier percentages.

4. **Outlier Analysis and Removal**
   - Calculated and printed the percentage of outliers in each column.
   - Removed rows where `MEDV` was at the cap (≥50.0) for improved prediction accuracy.

5. **Distribution Analysis**
   - Visualized each feature’s distribution to assess skewness:
     - Noted that `CRIM`, `ZN`, and `B` were highly skewed, indicating possible need for transformation.
     - Observed `MEDV` and other features had near-normal or bimodal distributions.

6. **Correlation Analysis**
   - Used a heatmap to examine feature correlations.
   - Selected features with correlation scores > 0.5 for `MEDV` as strong predictors, such as `LSTAT`, `INDUS`, `NOX`, `PTRATIO`, and `RM`.

7. **Feature Transformation and Scaling**
   - Applied `MinMaxScaler` to normalize predictor variables and visualized selected predictors against `MEDV`.
   - Performed log transformations on skewed features to reduce skewness and improve linearity.

8. **Data Splitting**
   - Defined features (`X`) and target variable (`y`), with `MEDV` as the dependent variable.
   - Split data into training and testing sets (80% train, 20% test) to enable model training and evaluation.

9. **Feature Scaling for Model Training**
   - Scaled `X_train` and `X_test` data using `StandardScaler` for optimal model performance in the regression model.

10. **Model Training**
    - Initialized and trained a `LinearRegression` model on the scaled training data using `model.fit()`.

11. **Prediction and Cross-Validation**
    - Used the trained model to predict `MEDV` on test data.
    - Calculated mean squared error (MSE) as a cross-validation metric.

12. **Model Evaluation Metrics**
    - Calculated `R-squared` and `Adjusted R-squared` values to assess model performance.
    - Printed the model’s coefficients and intercept to understand feature importance.

13. **Comparison of Actual vs. Predicted Values**
    - Created a DataFrame to compare actual `MEDV` values with predicted values and displayed the first ten rows.

14. **Visualizing Model Performance**
    - **Scatter Plot**: Compared actual vs. predicted values with a perfect prediction line to evaluate model accuracy visually.
    - **Residual Plot**: Plotted residuals (errors) against predictions to check for random distribution around zero, indicating model fit quality.
   
## Summary of Findings

1. **Model Performance and Accuracy**
   - **Cross-Validation Mean Squared Error (MSE)**: 8.1495
     - The relatively low MSE indicates a good fit, though some room for improvement exists.
   - **R-squared**: 0.8284
     - This score suggests the model explains approximately 82.8% of the variance in `MEDV`, which is a strong indication of model accuracy.
   - **Adjusted R-squared**: 0.8018
     - The adjusted R-squared is slightly lower than the R-squared value, accounting for the number of predictors in the model.
   
2. **Model Coefficients and Feature Impact**
   - The table below shows each feature’s coefficient in predicting `MEDV`.
   - **Interpretation**:
     - Positive coefficients indicate an increase in `MEDV` as the feature value increases, while negative coefficients indicate a decrease.
     - Notable high-impact features include:
       - **RM** (Rooms per dwelling): Positive coefficient (+2.5128), suggesting that more rooms per dwelling increase `MEDV`.
       - **PTRATIO** (Pupil-teacher ratio): Negative coefficient (-2.4339), suggesting that higher ratios are associated with lower `MEDV`.
       - **LSTAT** (Lower status of the population): Negative coefficient (-2.5557), indicating that higher `LSTAT` correlates with lower `MEDV`.

| Feature        | Coefficient |
|----------------|-------------|
| CRIM           | -0.8754     |
| ZN             | 0.7944      |
| INDUS          | -0.1957     |
| CHAS           | 0.0576      |
| NOX            | -1.6237     |
| RM             | 2.5128      |
| AGE            | -0.5803     |
| DIS            | -2.5656     |
| RAD            | 2.4216      |
| TAX            | -2.4339     |
| PTRATIO        | -1.9463     |
| B              | 0.8237      |
| LSTAT          | -2.5557     |
| **Intercept**  | 21.6681     |

3. **Comparison of Actual vs. Predicted Prices**
   - Observed minor deviations between actual and predicted values, suggesting the model has good prediction alignment.
   - The table below illustrates the comparison between actual and predicted `MEDV` values for a sample of test data.

| Index | Actual Values | Predicted Values |
|-------|---------------|------------------|
| 0     | 20.3          | 22.000          |
| 1     | 32.7          | 30.228          |
| 2     | 8.5           | 15.459          |
| 3     | 29.8          | 31.425          |
| 4     | 23.4          | 25.080          |
| 5     | 12.0          | 11.334          |
| 6     | 21.4          | 19.730          |
| 7     | 22.2          | 20.567          |
| 8     | 18.2          | 19.111          |
| 9     | 20.5          | 20.493          |

4. **Visual Representation of Predictions**
   - **Scatter Plot**:
     - A scatter plot of actual vs. predicted `MEDV` values highlights that most points are close to the line of perfect prediction, indicating the model is performing well.
   - **Residual Plot**:
     - Residuals (errors) plot shows a near-random distribution around zero, which suggests minimal bias and variance in the model’s predictions.

### Reasons why Random Forest and XGBoost often perform better than Linear Regression for the Boston housing dataset:

- **Captures Non-linear Relationships:** Random Forest and XGBoost can model complex, non-linear relationships, while Linear Regression assumes a linear relationship between input features and output.

- **Feature Interaction Handling:** These ensemble models can automatically capture interactions between features, which often exist in real-world data but aren't accounted for in linear regression without manual feature engineering.

- **Less Impact from Outliers:** Ensemble models are less sensitive to outliers in the data. Linear Regression, on the other hand, is highly influenced by outliers, which can skew predictions.

- **Robustness with Collinearity:** If input features are correlated (multicollinearity), Random Forest and XGBoost handle it better. Linear Regression is sensitive to multicollinearity, which can reduce its accuracy.

- **Better Performance on High-dimensional Data:** Both Random Forest and XGBoost tend to perform well even with high-dimensional and complex datasets due to their ensemble nature.

- **Automatic Feature Importance Evaluation:** These models can assess feature importance, helping in selecting the most influential features. Linear Regression requires separate statistical tests to determine feature importance.


## **Introduction to Logistic Regression**

- **Logistic regression** is used for **binary classification**, predicting probabilities of **categorical outcomes**.

## **Dataset for Logistic Regression (Pima Indian Diabetes)**

- Used for **binary classification** tasks, like predicting diabetes.
- **Features**:
  - **Pregnancies**: *Number of times the patient was pregnant.*
  - **Glucose**: *Plasma glucose concentration post-oral glucose tolerance test.*
  - **Blood Pressure**: *Diastolic blood pressure (mm Hg).*
  - **Skin Thickness**: *Triceps skin fold thickness (mm).*
  - **Insulin**: *Serum insulin levels post-test (mu U/ml).*
  - **BMI**: *Body mass index.*
  - **Diabetes Pedigree Function**: *Genetic predisposition score.*
  - **Age**: *Patient age.*
  - **Outcome**: *Diabetes presence (1) or absence (0).*

## **Project Objectives for Logistic Regression**

1. **Data Collection**:
   - Prepare and preprocess the **Pima Indian diabetes dataset**.

2. **Exploratory Data Analysis (EDA)**:
   - Identify patterns, correlations, and key features.

3. **Feature Selection**:
   - Select relevant features for dimensionality reduction.

4. **Model Development**:
   - Develop a **logistic regression model** for diabetes prediction.

5. **Model Evaluation**:
   - Evaluate using metrics such as **accuracy, precision, recall,** and **F1 score**.

6. **Result Interpretation**:
   - Analyze coefficients to interpret influence on diabetes risk.

7. **Reporting**:
   - Summarize methodology, findings, and insights for stakeholders.


## **Methodoloy**

1. **Importing Libraries**:
   - Imports essential libraries for data manipulation **(pandas)**, numerical operations **(numpy)**, and model evaluation.
   -  **scikit-learn**: Machine learning library for data splitting, scaling, modeling, and evaluation.
2. **Loading Dataset**:
   - Reads the diabetes dataset from a CSV file and displays the first 10 rows to give an overview of the data.
   - The **info()** method provides a summary of the dataset, including data types and non-null counts.
3. **Checking for Missing Values**:
   - Checks for any missing values in the dataset, which is crucial for data quality before training the model.
4. **Defining Features and Target Variable**:
   - **X** contains all the feature columns (inputs), while **y** contains the target variable (output), indicating diabetes presence.
5. **Data Split**:
   - Allocates 20% of the data for **testing** and 80% for **training**.
6. **Standardizing Features**:
   - This is used to standardize features by removing the mean and scaling to unit variance.
7. **Visualizing Feature Distributions Using Boxplots**:
   - To compare the distributions of various features in the Pima Indian Diabetes dataset with respect to the target variable **(Outcome)**, using boxplots to identify trends, outliers, and distribution differences.
8. **Visualizing Corelation using Heatmap**:
    - To analyze the correlation between different features in the Pima Indian Diabetes dataset, helping to identify relationships and potential predictors for the target variable **(Outcome)**.
9. **Visualizing Relationships Between Features Using Pairplot**:
    - To explore the relationships between different features in the Pima Indian Diabetes dataset and understand how they relate to the target variable **(Outcome)** using a pairplot.
10. **Creating a Logistic Regression Model**:
    - Created and assigned to the variable model.
11. **Training the Model**:
    - Using the training dataset **X_train** (features) and **y_train** (target variable).
12. **Making Predictions**:
    - Standardize test data, then predict class labels using trained model.
13. **Prediction for a Specific Input**:
    - Predict outcome using logistic regression model.
14. **Calculating Accuracy**:
    - Measures how accurate those predictions are compared to the true labels.
15. **Visualizing Predicted vs Actual Values**:
    - To compare the predicted outcomes from a classification model with the actual outcomes in the Pima Indian Diabetes dataset, using a count plot for clear visual representation.
16. **Importing Confusion Matrix**:
    - This code creates a confusion matrix to visualize model performance by comparing actual and predicted outcomes.
17. **Calculating Precision**:
    - Precision indicates how many of the predicted positive cases were actually positive.
18. **Calculating F1 Score**:
    - Providing a balance between the two metrics.
19. **Calculating the Recall**:
    - Indicating how effectively the model identifies positive cases from the actual positives.

## **Interpretation**
**1.  Model's Ability to Classify**
 - **Classification Performance:** The logistic regression model is used to classify individuals as diabetic or non-diabetic based on various health metrics. The effectiveness of this model can be assessed through several metrics:
  - **Accuracy:** An accuracy score around 82% indicates that the model is reasonably effective in distinguishing between diabetic and non-diabetic individuals.
  - **Precision:** The precision score indicates the proportion of true positive predictions out of all positive predictions made by the model.
  - **Recall:** Recall measures the model's ability to identify actual positive cases.

**2. Importance of Features**
 - **Feature Coefficients:** In logistic regression, the model assigns a coefficient to each feature, representing its influence on the probability of having diabetes:
  - **Positive Coefficients:** Features with positive coefficients increase the odds of diabetes. For example:
    - **Glucose Level:** Typically has a strong positive coefficient, indicating that higher glucose levels significantly elevate diabetes risk.
    - **Body Mass Index (BMI):** Generally shows a positive relationship, where increased BMI correlates with higher diabetes likelihood, reflecting obesity's role as a risk factor.
  - **Negative Coefficients:** Features with negative coefficients indicate a decrease in the odds of diabetes as the feature value increases. Understanding these relationships helps identify protective factors.
- **Key Features Influencing Diabetes:**
  - **Glucose:** Often the most critical predictor, as high glucose levels directly indicate diabetes risk.
  - **BMI:** Important for assessing obesity-related risks.
  - **Age:** Typically shows a positive correlation, with older individuals at greater risk.
  - **Insulin Levels and Blood Pressure:** These features also contribute significantly to diabetes prediction, indicating the model's capacity to incorporate a range of health metrics.
    

---

## **References**

1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python*. [Link](https://scikit-learn.org/stable/index.html)
2. McKinney, W. (2010). *Data structures for statistical computing in Python*. [Link](https://pandas.pydata.org/pandas-docs/stable/)
3. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. [Link](https://matplotlib.org/stable/users/index.html)
4. UCI Machine Learning Repository. *Boston Housing Data*. [Link](https://archive.ics.uci.edu/ml/datasets/Housing)
5. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.[Link](https://doi.org/10.1023/A:1010933404324)
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794..[Link](https://doi.org/10.1023/A:1010933404324)
7. Chang, V., Bailey, J., Xu, Q.A. et al. (2022). *Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms.* [Link](https://doi.org/10.1007/s00521-022-07049-z)
8. shrutimechlearn. *Step by step diabetes classification*. [Link](https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification)
9. Vincent Lugat. (2019) *Pima Indians Diabetes - EDA & Prediction* [Link](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)
