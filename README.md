# Midterm Project for Elective 2
## Table of Contents

1. [Abstract of the Project](#abstract-of-the-project)
2. [Project Objectives](#project-objectives)
3. [Linear Regression](#linear-regression)
   - [A. What is Linear Regression?](#a-what-is-linear-regression)
   - [B. Overview of Linear Regression for the Boston Housing Dataset](#b-overview-of-linear-regression-for-the-boston-housing-dataset)
   - [C. Dataset Description](#c-dataset-description)
   - [D. Why MEDV Should Be the Dependent Variable](#d-why-medv-should-be-the-dependent-variable)
   - [E. Data Type of MEDV](#e-data-type-of-medv)
   - [F. Appropriateness for Linear Regression](#f-appropriateness-for-linear-regression)
   - [G. Initial Observation of the Dataset](#g-initial-observation-of-the-dataset)
   - [H. Methodology](#h-methodology)
   - [I. Summary of Findings](#i-summary-of-findings)
   - [J. Why Other Models Perform Better Than Linear Regression](#j-why-other-models-perform-better-than-linear-regression)
4. [Logistic Regression](#logistic-regression)
   - [A. What is Logistic Regression?](#a-what-is-logistic-regression)
   - [B. Dataset for Logistic Regression (Pima Indian Diabetes)](#b-dataset-for-logistic-regression-pima-indian-diabetes)
   - [C. Why "Outcome" (Presence of Diabetes) Should Be the Dependent Variable](#c-why-outcome-presence-of-diabetes-should-be-the-dependent-variable)
   - [D. Data Type of the "Outcome" Variable](#d-data-type-of-the-outcome-variable)
   - [E. Appropriateness for Logistic Regression](#e-appropriateness-for-logistic-regression)
   - [F. Initial Observations of the Dataset](#f-initial-observations-of-the-dataset)
   - [G. Methodology](#g-methodology)
   - [H. Summary of Findings](#h-summary-of-findings)
5. [Discussion of Regression Analysis Results](#discussion-of-regression-analysis-results)
   - [A. Reflection on Results](#a-reflection-on-results)
   - [B. Comparison of the Two Regression Methods](#b-comparison-of-the-two-regression-methods)
   - [C. Limitations](#c-limitations)
6. [References](#references)
7. [Group Members](#group-members)

## Abstract of the Project
1. **Project Overview**:
   - This project explores predictive modeling using linear and logistic regression techniques to analyze real-world data.
   - Focuses on two datasets:
     - **Boston Housing**: Examines socioeconomic and environmental factors impacting housing prices.
     - **Pima Indian Diabetes**: Assesses health-related factors influencing diabetes prevalence.

2. **Objective**:
   - Demonstrate regression applications in predicting continuous and categorical outcomes.
   - Provide insights into factors that influence housing prices and diabetes likelihood.

3. **Linear Regression Analysis**:
   - **Dataset**: Boston Housing dataset.
   - **Goal**: Predict median home values based on variables such as crime rates, proximity to highways, and pollution levels.
   - **Data Preprocessing**:
     - Outlier handling, normalization, and feature transformation were applied to improve model accuracy.
   - **Model Performance**:
     - Achieved an R-squared value of 0.8284, indicating a strong model fit.
     - Challenges: Potential non-linear relationships and multicollinearity among features.

4. **Logistic Regression Analysis**:
   - **Dataset**: Pima Indian Diabetes dataset.
   - **Goal**: Classify individuals as diabetic or non-diabetic based on factors like glucose levels, BMI, and age.
   - **Data Preprocessing**:
     - Included categorical encoding, class balancing, and scaling to improve classification accuracy.
   - **Model Performance**:
     - Achieved around 82% accuracy.
     - Metrics such as precision and recall were used to assess the classification performance.

5. **Key Insights and Limitations**:
   - Both regression models are useful in different contexts:
     - **Linear Regression**: Effective in capturing relationships in housing prices.
     - **Logistic Regression**: Valuable in predicting diabetes likelihood.
   - **Limitations**:
     - Sensitivity to assumptions of linearity, outliers, and complex feature interactions.
     - Suggested improvements include using ensemble methods (e.g., Random Forest, XGBoost) for capturing non-linear relationships.

6. **Documentation**:
   - Project methodology, results, and findings are documented in a structured GitHub repository.
   - Provides a comprehensive resource on regression techniques and their applications in predictive analytics.

## Project Objectives

1. **Understand the Dataset and Select Appropriate Variables**
   - Explore and analyze the assigned dataset to understand its structure and variables.
   - Identify suitable dependent variables for:
     - **Linear Regression**: A continuous variable.
     - **Logistic Regression**: A categorical variable.
   - Justify the choice of dependent variables based on dataset characteristics and the objectives of each regression analysis.

2. **Apply Regression Techniques to Real-World Data**
   - Conduct a **Linear Regression Analysis** using the identified continuous variable.
     - Preprocess data (handle missing values, outliers, normalization) to improve model accuracy.
     - Train and evaluate the model, utilizing metrics such as R-squared and Mean Squared Error.
     - Interpret the model coefficients to understand the relationship between dependent and independent variables.
   - Conduct a **Logistic Regression Analysis** using the identified categorical variable.
     - Preprocess data (categorical encoding, class balancing) to improve classification accuracy.
     - Train and evaluate the model, utilizing metrics such as accuracy and confusion matrix.
     - Discuss the model’s classification performance and feature importance.

3. **Document and Share Findings through GitHub**
   - Set up a well-structured GitHub repository to document the project, including:
     - A clear overview of regression concepts and dataset description.
     - Detailed steps for data preprocessing, model implementation, and analysis.
     - Summaries of findings, interpretations of model results, and any observed limitations.

4. **Present Key Insights and Reflections**
   - Provide a clear and organized presentation on:
     - The methodology applied in both Linear and Logistic Regression.
     - Key findings and insights gained from the analyses.
     - A reflection on the strengths and limitations of each regression approach as applied to the dataset.



## **Linear Regression** 

### A. What is Linear Regression?

- **Linear regression** is a fundamental algorithm in *supervised machine learning*.
- Used to model relationships between a **dependent variable** (*target*) and one or more **independent variables** (*features*).
- In *simple linear regression*, a single feature is used to predict a continuous output variable.
- *Multiple linear regression* uses several features to predict a continuous output.
- Widely applied across fields like **economics**, **biology**, and **real estate** to analyze and predict trends.
- In this project, linear regression will predict the **median value of homes** in various **Boston suburbs** based on socio-economic and environmental factors.

### B. Overview of Linear Regression for the Boston Housing Dataset

- **Goal**: Predict the **median value of owner-occupied homes** (*target variable: MEDV*) based on factors such as:
  - **CRIM** - *Crime rate per capita*
  - **RM** - *Average number of rooms per dwelling*
  - **RAD** - *Proximity to highways*
- Key questions addressed include:
  - How environmental factors like **air quality** (*NOX*) or **proximity to Charles River** (*CHAS*) impact home values.
  - Whether **property taxes** (*TAX*) and **pupil-teacher ratios** (*PTRATIO*) correlate with lower housing prices.
  - If socio-economic factors, such as **crime rate** (*CRIM*) or **percentage of lower-status residents** (*LSTAT*), explain fluctuations in real estate prices.

### C. Dataset Description

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
 
### D. Why MEDV Should Be the Dependent Variable
- `MEDV` represents the median home price, which is typically the main target for prediction in real estate datasets.
- Predicting `MEDV` can provide insights into housing market trends, affordability, and property valuations in Boston’s suburbs.
- Most features in this dataset (e.g., crime rate, number of rooms, accessibility to highways) influence home values, making `MEDV` an appropriate target for regression analysis.

### E. Data Type of MEDV
- `MEDV` is **continuous**, as it represents home prices, which can take a range of numerical values, not limited to categories or fixed intervals.

### F. Appropriateness for Linear Regression
- Since `MEDV` is a continuous variable, it is suitable for linear regression, which predicts continuous outcomes.
- Linear regression assumes a linear relationship between the target and predictor variables. This can help model how various factors like room count or crime rate contribute to home price fluctuations.

### G. Initial Observation of the Dataset

- **Potential Non-Linear Relationships:** Some features (e.g., crime rate) may have a non-linear relationship with `MEDV`, which might affect linear regression's performance.
- **Outliers in `MEDV`:** The dataset may contain outliers in home prices, which could skew predictions and reduce model accuracy.
- **Feature Multicollinearity:** There might be correlations among features (e.g., `NOX` and `INDUS`), which could introduce multicollinearity issues.
- **Skewed Distributions:** Certain variables may have skewed distributions, potentially influencing regression results and suggesting the need for feature transformation or regularization.

### H. Methodology

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
   
### I. Summary of Findings

1. **Model Performance Metrics**
   - **Cross-Validation Mean Squared Error (MSE)**: 8.1495
     - This value quantifies the average squared differences between the predicted and actual `MEDV` values. The lower the MSE, the better the model fits the data.
     - Here, an MSE of 8.1495 suggests that the model is relatively accurate, though there is still some prediction error, potentially due to noise or non-linear relationships in the data.
   - **R-squared (R²)**: 0.8284
     - The R² value shows the proportion of variance in `MEDV` that is explained by the model’s features.
     - An R² of 0.8284 means that about 82.8% of the variance in housing prices is captured by this model, indicating a strong fit.
   - **Adjusted R-squared**: 0.8018
     - The Adjusted R² compensates for the number of predictors, adjusting R² downward if unnecessary variables are included.
     - Here, an adjusted R² of 0.8018 suggests that most predictors are relevant, as there is only a slight reduction from the standard R².

2. **Feature Impact Analysis (Model Coefficients)**
   - Coefficients represent the impact of each feature on `MEDV`. A positive coefficient implies that an increase in the feature value will increase `MEDV`, while a negative coefficient implies the opposite.
   - The table below details each feature’s coefficient:

| Feature        | Coefficient | Interpretation                                                                                     |
|----------------|-------------|-----------------------------------------------------------------------------------------------------|
| **CRIM**       | -0.8754     | Higher crime rates are associated with lower housing prices, possibly due to reduced neighborhood desirability. |
| **ZN**         | 0.7944      | Residential land zoning for larger lots positively affects housing prices, suggesting exclusivity adds value.  |
| **INDUS**      | -0.1957     | Higher industrial activity slightly decreases housing prices, potentially due to increased pollution or noise. |
| **CHAS**       | 0.0576      | Proximity to the Charles River has a slight positive impact on housing prices, adding scenic or recreational value. |
| **NOX**        | -1.6237     | Nitric oxide concentration, indicating pollution, negatively impacts housing prices significantly. |
| **RM**         | 2.5128      | Average number of rooms per dwelling has the strongest positive impact, showing that larger homes are valued higher. |
| **AGE**        | -0.5803     | Older properties are slightly less valuable, potentially due to wear or less modern features. |
| **DIS**        | -2.5656     | Greater distance from employment centers decreases prices, indicating a preference for proximity to work. |
| **RAD**        | 2.4216      | Accessibility to highways slightly increases prices, suggesting improved commute convenience. |
| **TAX**        | -2.4339     | Higher property tax rates are associated with lower prices, as it increases the cost of owning property. |
| **PTRATIO**    | -1.9463     | Higher student-teacher ratios negatively impact housing prices, likely reflecting school quality perceptions. |
| **B**          | 0.8237      | A higher proportion of African American residents slightly increases housing prices.              |
| **LSTAT**      | -2.5557     | Higher percentage of lower-status individuals significantly decreases prices, reflecting socioeconomic impact. |
| **Intercept**  | 21.6681     | Baseline housing price when all predictors are zero; not directly interpretable due to lack of practical context. |


3. **Comparison of Actual vs. Predicted Prices**
   - The table below provides a comparison of actual and predicted `MEDV` values, showing the model’s accuracy in specific instances.

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

   - **Interpretation**:
     - Most predicted values closely align with actual values, suggesting that the model captures important relationships in the data.
     - Some discrepancies exist (e.g., index 2), highlighting that certain data points may have additional influencing factors or contain noise.

4. **Visual Analysis of Predictions**
   - **Scatter Plot of Actual vs. Predicted Values**:
     - Points closely following the red diagonal line (indicating perfect prediction) show high prediction accuracy.
     - Deviation from the line suggests areas where the model could improve, such as capturing non-linear relationships or additional predictors.
     
 <div align="center">
   
  ![ScatterPlot](https://github.com/user-attachments/assets/1f994071-4f1e-46a7-a4df-30d5683de8ab)
    
</div>

   - **Residual Plot**:
     - The residual plot shows the distribution of errors (differences between actual and predicted values).
     - A random, symmetric distribution around zero suggests that the model has low bias, meaning it performs consistently across different values of `MEDV`
     
<div align="center">    
  
![Residual Plot](https://github.com/user-attachments/assets/5f25354e-f559-4af9-8d4f-f94ea184a5a0)

</div>

### J. Why Other Models Perform Better Than Linear Regression

#### Reasons why Random Forest and XGBoost often perform better than Linear Regression for the Boston housing dataset:

- **Captures Non-linear Relationships:** Random Forest and XGBoost can model complex, non-linear relationships, while Linear Regression assumes a linear relationship between input features and output.

- **Feature Interaction Handling:** These ensemble models can automatically capture interactions between features, which often exist in real-world data but aren't accounted for in linear regression without manual feature engineering.

- **Less Impact from Outliers:** Ensemble models are less sensitive to outliers in the data. Linear Regression, on the other hand, is highly influenced by outliers, which can skew predictions.

- **Robustness with Collinearity:** If input features are correlated (multicollinearity), Random Forest and XGBoost handle it better. Linear Regression is sensitive to multicollinearity, which can reduce its accuracy.

- **Better Performance on High-dimensional Data:** Both Random Forest and XGBoost tend to perform well even with high-dimensional and complex datasets due to their ensemble nature.

- **Automatic Feature Importance Evaluation:** These models can assess feature importance, helping in selecting the most influential features. Linear Regression requires separate statistical tests to determine feature importance.


## **Logistic Regression**

### A. What is Logistic Regression?

- **Logistic regression** is a popular algorithm in *supervised machine learning* primarily used for **classification tasks**.
- Models the probability that a given input belongs to a particular **class or category**.
- Useful when the **dependent variable** is **binary** (e.g., 0 or 1, Yes or No, True or False).
- The algorithm estimates the **likelihood** of an event happening by fitting data to a **logistic function** (also called a sigmoid function).
- Extends to **multiclass classification** through methods like *One-vs-Rest (OvR)* or *Softmax Regression*.
- Commonly applied in fields like **medicine** (e.g., disease prediction), **finance** (e.g., loan approval), and **marketing** (e.g., customer churn prediction).
- In logistic regression, instead of predicting exact values, the model predicts **probabilities** and maps them to classes based on a **decision threshold** (usually 0.5).


### B. Dataset for Logistic Regression (Pima Indian Diabetes)

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
 
### C. Why "Outcome" (Presence of Diabetes) Should Be the Dependent Variable
  - The "Outcome" column represents whether or not an individual has diabetes, coded as:
    - `1` for diabetic
    - `0` for non-diabetic
  - Since the goal of the analysis is to predict diabetes presence based on other factors, "Outcome" naturally serves as the dependent variable.

### D. Data Type of the "Outcome" Variable
  - **Categorical**
  - The variable "Outcome" is binary, representing two distinct categories (diabetic and non-diabetic).

### E. Appropriateness for Logistic Regression
  - Logistic regression is well-suited for binary outcomes, as it estimates the probability of an observation belonging to one of two categories.
  - The logistic function constrains output to a range between `0` and `1`, making it ideal for probability-based classification.

### F. Initial Observations of the Dataset
  - **Potential Issues:**
    - Several columns (e.g., `Insulin` and `SkinThickness`) have zeros where actual values should be present, possibly indicating missing data.
    - Zero values for attributes like `BloodPressure` and `BMI` may also indicate data entry issues, as these values are biologically implausible.
  - **Patterns:**
    - Columns such as `Pregnancies`, `BMI`, `Age`, and `Glucose` could show a correlation with the "Outcome" variable, which logistic regression can help identify.
  - **Size and Balance:**
    - The dataset contains 768 entries, which is moderately sized for logistic regression analysis.
    - It's useful to further examine the balance between classes (i.e., counts of `0` vs. `1` in "Outcome") for potential imbalance, which could affect model performance.


### G. Methodology

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

### H. Summary of Findings
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

**3. Countplot Visualization**:
 <div align="center">
   
![Countplot logi](https://github.com/user-attachments/assets/994e8ef6-7728-4f28-a742-3f8ffb470be4)

 </div>
 
 - **Interpretation:**

  - **No Diabetes Predictions:**
    - The model accurately predicts a high number of "No Diabetes" cases, showing strong performance in identifying individuals without diabetes.
  - **Diabetes Predictions:**
    - While the count of predicted "Diabetes" cases is slightly lower than actual cases, the model shows promising initial performance, capturing many diabetic cases accurately.
  - **Balanced Insights:**
    - The plot highlights both the model's strengths and areas for fine-tuning, providing a solid foundation for further improvements.
  
- **Implications for Model Development:**
  - The model's accuracy in predicting "No Diabetes" suggests it is effectively identifying non-diabetic individuals.
  - With minor adjustments, such as enhancing sensitivity for diabetic cases, the model has the potential to provide even more accurate and reliable predictions across both categories.

---
    
**4. Confusion Matrix**:

 <div align="center">
   
![Confu mat logi](https://github.com/user-attachments/assets/55b981d4-c72b-4ca3-b34f-c48f557a409c)

 </div>
---

 - **Interpretation:**
- **True Negatives (Correct No Diabetes Predictions):**
    - The model correctly identifies **98 non-diabetic cases**, demonstrating strong accuracy in recognizing individuals without diabetes.
  - **True Positives (Correct Diabetes Predictions):**
    - The model accurately predicts **29 diabetic cases**, showing a good foundation for detecting diabetes.
  - **False Positives and False Negatives:**
    - With only **9 False Positives** and **18 False Negatives**, the model maintains a relatively low error rate, showcasing its reliability.
    - These manageable error counts provide a basis for further refinement, allowing for even better performance in future iterations.

- **Model Strengths and Future Enhancements:**
  - **Accuracy in Non-Diabetic Predictions:** The high number of True Negatives indicates the model is highly effective in predicting "No Diabetes" cases.
  - **Balanced Detection of Diabetic Cases:** The model performs well in identifying diabetic cases, and minor adjustments could further enhance its sensitivity and recall.
  - **Room for Growth:** The matrix highlights the model’s current strengths, while also pointing to opportunities for fine-tuning, particularly to improve sensitivity for detecting diabetes cases.
  - **Overall Performance:** This initial model demonstrates a solid understanding of diabetes prediction, with a promising base that can be incrementally improved for even greater accuracy.
 
## Discussion of Regression Analysis Results

### A. Reflection on Results
- **Linear Regression**: 
    - The linear regression model achieved a **Mean Squared Error (MSE) of 8.1495**, suggesting a generally accurate fit. However, the presence of some prediction errors, possibly due to data noise or underlying non-linear relationships, indicates room for improvement.
    - An **R-squared value of 0.8284** reveals that the model explains 82.8% of the variance in housing prices, which indicates that it captures most key relationships in the data. 
    - The **Adjusted R-squared value of 0.8018** supports that most predictors are relevant to the target variable, **Median House Value (MEDV)**, by showing only a slight reduction from the R-squared value.
    - **Feature Impact Analysis** revealed that features like **Crime Rate (CRIM)** and **Nitric Oxide Concentration (NOX)** negatively impact housing prices, while features such as the **Average Number of Rooms (RM)** and **Accessibility to Highways (RAD)** positively influence prices.

- **Logistic Regression**:
    - The logistic regression model achieved a **classification accuracy of around 82%**, effectively distinguishing between diabetic and non-diabetic individuals.
    - **Feature Coefficients Analysis** showed that **Glucose Level** and **Body Mass Index (BMI)** have strong positive coefficients, indicating that higher levels of these features increase the likelihood of diabetes. Other features, such as **Age** and **Insulin Levels**, also significantly impact diabetes risk.

### B. Comparison of the Two Regression Methods
- **Purpose of Each Model**:
    - **Linear Regression** is used to predict a continuous outcome (housing prices). It aims to find a straight-line relationship between the predictor variables and the target variable, so that the model's predictions closely match the actual values.
    - **Logistic Regression** is used for classification tasks, predicting a binary outcome (diabetic or non-diabetic). Instead of finding a line, logistic regression produces an "S-shaped" curve that estimates the probability of each class (in this case, the probability of being diabetic).

- **Model Performance**:
    - The **MSE and R-squared** in the linear regression model demonstrate how well the model predicts precise values of housing prices, while **accuracy, precision, and recall** in the logistic regression model measure how well it classifies individuals as diabetic or non-diabetic.
    - Linear regression results show a relatively strong fit to the data, while logistic regression’s accuracy of 82% suggests it is fairly effective for classification.

- **Feature Impact**:
    - Both models analyze the importance of features by assigning coefficients, but these coefficients reflect different outcomes. In linear regression, a positive coefficient indicates that as a feature value increases, the target value (housing price) increases as well. In logistic regression, a positive coefficient indicates that as a feature value increases, the probability of being diabetic also increases.

### C. Limitations
- **Linear Regression Limitations**:
    - Linear regression assumes a **linear relationship** between the predictors and the target variable. If the true relationship is non-linear, as may be the case with housing prices, the model may not fully capture all the nuances, leading to residual errors.
    - The **model is sensitive to outliers**, which can skew predictions and reduce accuracy, particularly when extreme values exist within the data.
    - **Multicollinearity** among features (where predictors are correlated with each other) can also affect the reliability of coefficients, leading to instability in the model’s predictions.

- **Logistic Regression Limitations**:
    - Logistic regression assumes a **linear relationship between predictors and the log odds** of the outcome, which may oversimplify complex relationships between health metrics and diabetes risk.
    - Similar to linear regression, **outliers** can affect model performance, and logistic regression may also be influenced by **imbalanced data**, where one outcome (e.g., diabetic or non-diabetic) is more prevalent. This can skew the model’s accuracy, as it may perform better on the more frequent class.
    - Logistic regression does not naturally handle **non-linear relationships** and may benefit from additional data preprocessing or transformation if non-linear patterns exist in the data.

Overall, while both models provide useful insights and relatively strong performance, limitations such as sensitivity to linear assumptions, outliers, and complex relationships could potentially impact their predictive power. Addressing these limitations through additional model adjustments or using more advanced algorithms could further improve predictive accuracy.


## References

1. Pedregosa, F. et al. (2011). *Scikit-learn: Machine learning in Python*. [Link](https://scikit-learn.org/stable/index.html)
2. McKinney, W. (2010). *Data structures for statistical computing in Python*. [Link](https://pandas.pydata.org/pandas-docs/stable/)
3. Hunter, J. D. (2007). *Matplotlib: A 2D graphics environment*. [Link](https://matplotlib.org/stable/users/index.html)
4. UCI Machine Learning Repository. *Boston Housing Data*. [Link](https://archive.ics.uci.edu/ml/datasets/Housing)
5. Breiman, L. (2001). Random forests. Machine Learning, 45(1), 5-32.[Link](https://doi.org/10.1023/A:1010933404324)
6. Chen, T., & Guestrin, C. (2016). XGBoost: A scalable tree boosting system. Proceedings of the 22nd ACM SIGKDD International Conference on Knowledge Discovery and Data Mining, 785-794..[Link](https://doi.org/10.1023/A:1010933404324)
7. Chang, V., Bailey, J., Xu, Q.A. et al. (2022). *Pima Indians diabetes mellitus classification based on machine learning (ML) algorithms.* [Link](https://doi.org/10.1007/s00521-022-07049-z)
8. shrutimechlearn. *Step by step diabetes classification*. [Link](https://www.kaggle.com/code/shrutimechlearn/step-by-step-diabetes-classification)
9. Vincent Lugat. (2019) *Pima Indians Diabetes - EDA & Prediction* [Link](https://www.kaggle.com/code/vincentlugat/pima-indians-diabetes-eda-prediction-0-906)

## Group Members
1. Evangelista, Lexter Jhustin L. (20-60481)
2. Malabanan, Angelo Louis D. (21-04184)
