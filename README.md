# Midterm Project for Elective 2

## **Introduction**

### **Overview of Linear Regression**

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

## **Project Objectives**

1. **Feature Impact on Housing Prices**:
   - Identify which **features significantly impact** housing prices.
2. **Model Building**:
   - Build a regression model to predict **housing prices** using socio-economic factors.
3. **Model Evaluation**:
   - Evaluate using **R-squared** and **RMSE** to assess model performance.
4. **Residual Analysis**:
   - Analyze residuals to evaluate accuracy and identify improvement areas.

#### **Model's Ability to Predict (R² Value = 0.6688)**

- **R² = 0.6688**: Model explains ~66.88% of the **variance in median home values**.
  - **Prediction Comparison**:
    - Example predictions show deviations, like:
      - **Actual**: 23.6, **Predicted**: 28.99 (overestimation)
      - **Actual**: 13.6, **Predicted**: 14.82 (close prediction)

#### **Cross-Validation Mean Squared Error (MSE = 24.2911)**

- **MSE of 24.2911** suggests that, on average, the squared difference between predicted and actual home values is 24.2911 units squared.

#### **Interpreting Coefficients**

- Positive coefficients mean that **increases in the feature value** are associated with **higher predicted home values**.
- Negative coefficients mean that **increases in the feature value** lead to **lower predicted home values**.

## **Methodology**

1. **Importing Libraries**:
   - **pandas, numpy**: Data handling.
   - **matplotlib, seaborn**: Data visualization.
   - **scikit-learn**: Machine learning library for data splitting, scaling, modeling, and evaluation.

2. **Loading Dataset**:
   - Verify data integrity.
   
3. **Separating Features (X) and Target (y)**:
   - Define **X** (features) and **y** (target).

4. **Data Split**:
   - **80% Training**, **20% Testing**.

5. **Standardization**:
   - Use **StandardScaler** to normalize feature values.

6. **Model Training**:
   - Fit **Linear Regression** model to training data.

7. **Prediction and Evaluation**:
   - Assess using **Mean Squared Error (MSE)** and **R-squared (R²)**.

8. **Residual Analysis**:
   - Check residuals for prediction accuracy and potential model improvements.

9. **Visualization**:
   - Plot *Actual vs. Predicted values* and *Residuals*.

10. **Prediction on New Data**:
   - Use the trained model on new inputs for home value prediction.

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
   - **pandas,numpy**: to access various functionalities.
   -  **scikit-learn**: Machine learning library for data splitting, scaling, modeling, and evaluation.
2. **Loading Dataset**:
   - Use **pd.read_csv()** for loading CSV files into a Pandas DataFrame, which can then be manipulated and analyzed easily.
3. **Selecting Features (X):** and **Selecting Target Variable (y):**
   - **X:** Contains the feature data (independent variables) that will be used to train the model.
   - **y:** Contains the target variable (dependent variable) that the model will attempt to predict.
4. **Data Split**:
   - Allocates 20% of the data for **testing** and 80% for **training**.
5. **Standardizing Features**:
   - This line imports the **StandardScaler** class, which is used to standardize features by removing the mean and scaling to unit variance.
6.  **Creating a Logistic Regression Model**:
   - Created and assigned to the variable model.
7. **Training the Model**:
   - Using the training dataset **X_train** (features) and **y_train** (target variable)
---

## **References**

1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python*. [Link](https://scikit-learn.org/stable/index.html)
2. McKinney, W. (2010). *Data structures for statistical computing in Python*. [Link](https://pandas.pydata.org/pandas-docs/stable/)
3. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. [Link](https://matplotlib.org/stable/users/index.html)
4. UCI Machine Learning Repository. *Boston Housing Data*. [Link](https://archive.ics.uci.edu/ml/datasets/Housing)
5. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.[Link](https://doi.org/10.1023/A:1010933404324)
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794..[Link](https://doi.org/10.1023/A:1010933404324)
7. Chang, V., Bailey, J., Xu, Q.A. et al. (2022). *Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms.* [Link](https://doi.org/10.1007/s00521-022-07049-z) 
