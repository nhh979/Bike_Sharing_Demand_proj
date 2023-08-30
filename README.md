# Bike_Sharing_Demand_proj
# Table of Contents
[Problem Statement](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#i-problem-statement)
[](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#ii-dataset)
[](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#iii-data-pipeline)
[](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#iv-eda-summary)
[](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#v-model-performance)
[](https://github.com/nhh979/Bike_Sharing_Demand_proj/tree/main#vi-conclusion)

## I. Problem Statement
This is a Kaggle competition asking participants to build a model to predict bike rental demand in the Capital Bikeshare program in Washington, D.C., using the data of the first 19 days of each month as the training set while the test set contains the rest of each month.

## II. Dataset
- The dataset contains a training set and a test set. The training set includes 10886 observations and 12 attributes, wheares these number are 6493 and 9 respectively for the test set. More information about the competition and the dataset can be found [here](https://www.kaggle.com/competitions/bike-sharing-demand/data).  

**Data Fields:**

- **datetime** - hourly date + timestamp  
- **season** -  1 = spring, 2 = summer, 3 = fall, 4 = winter 
- **holiday** - whether the day is considered a holiday
- **workingday** - whether the day is neither a weekend nor holiday
- **weather** - 1: Clear, Few clouds, Partly cloudy, Partly cloudy  
2: Mist + Cloudy, Mist + Broken clouds, Mist + Few clouds, Mist  
3: Light Snow, Light Rain + Thunderstorm + Scattered clouds, Light Rain + Scattered clouds  
4: Heavy Rain + Ice Pallets + Thunderstorm + Mist, Snow + Fog   
- **temp**- temperature in Celsius
- **atemp** - "feels like" temperature in Celsius
- **humidity** - relative humidity
- **windspeed** - wind speed
- **casual** - number of non-registered user rentals initiated
- **registered** - number of registered user rentals initiated
- **count** - number of total rentals

## III. Data Pipeline
**1. Analyze Data:** We tried to understand some characteristics of the dataset such as its dimension, missing values, duplicated rows, and data type of each variable. We also wanted to look at some statistics summary to have a quick understanding of the dataset.  

**2. Exploratory Data Analysis (EDA):** This is a crucial process that gives us a deeper understanding of the dataset. In this step, We tried to find some interesting insights, identify patterns, and discover the relationships and trends between variables. This step is extremely helpful when it comes to detecting outliers, missing values, or any other issues that might affect our model building later.  

**3. Data Cleaning:** In this step, we removed outlying values in the target variables using the z-score of 3 method. We also determined that `humidity` and `windspeed` columns contains missing values represented by 0. Of those columns, `windspeed` has way too many missing values compared with `humidity` column (12% vs 0.2%)  

**4. Feature Engineering:** We created dummy variables for categorical variable such as `weather` and `season`. We also created new variables called `rush_hour` based on the hour of the day and categorical `temp` column based on the temperature during that time. Besides, we extracted time of day based on `datetime` column. We then dropped unnecessary features like `datetime`, `atemp` which is highly correlated with `temp`, `windspeed` which has a lot of missing values and has no correlation with the target variables, and `casual` and `registered` which do not appear in the test set.  

**5. Evaluation Metric:** The required evaluation metric is the Root Mean Square Logarithmic Error (RMSLE), defined by the following equation:

$$RMSLE = \sqrt{\frac{1}{n}\sum_{i=1}^{n}(log(p_{i} + 1) - log(a_{i}+1))^2}$$  

where $p_{i}$ is the predicted values and $a_{i}$ is the actual values.  

RMSLE scorer is utilized to assess the model's performance. RMSLE is a metric commonly used in regression tasks to measure the accuracy of predictions. It penalizes underestimation and overestimation of the target variable, making it suitable for this bike rental count prediction task. The lower the RMSLE value, the better the model's predictions align with the actual target values.

**6. Model Training:** We tried to build some different models such as Linear Regression, Lasso, Ridge, RandomForestRegressor and XGBRegressor. Later, we did hyperparameter tuning using GridSearchCV and RandomizedSearchCV to find the best estimator.  

**7. Model Evaluating:** Comparing the RMSLE scores between those models, we chose the final model that made better predictions with low RMSLE score.

## IV. EDA Summary
- The number of each season are pretty much similar. The number of non-holiday days dominates the number of holidays, and the number of workingday is about twice higher than the number of days off. These observations seem reasonable because the dataset time range from the beginning of 2011 to the end of 2012, which is about two entire years. However, the number of days that had storm are very few which is very unusual.
- Majority of rental count occured in nice weather. Although the number of days with storm is very few to none, the rental count on storm days was still very high on average in comparison with other types of weather.
- The total bike rental count is greatest in the fall as well as the average rental count, following by the summer, the winter and the spring respectively.
- The average rental count in 2012 was almost twice higher than that of in 2011.
- The highest average rental count took place from June to October.
- The average rent count was about the same in terms of day of month and day of week.
- The highest average rental count occured in the morning around 7-9AM, and in the evening around 17-18PM, during a day. 
- Registered bike rentals mostly utilize bikes during working days, which can be seen easily at 7AM and 5PM. This can be attributed to regular school and office commuters who rely on bicycles as their primary mode of transportation during these times.
- In contrast, non-registered bike rentals mostly utilize bike during non-working day and tend to rent bikes later during the day, around 10AM to 4PM. This trend can be explained by people using bike rentals for their leisure activities during this time of the day.
- There is a strong correlation between `temp` and `atemp`.Besides, there is no multicollinearity between other variables.
- Welch's t-test shows that there is no significant difference in average rental count between working days and non-working days, the same for holidays vs. regular days.

## V. Model Performance
- Random Forest and XGBoost outperform three other regression models as the RMSLE on both the cleaned and raw test sets of those are about tripple less than those of other model. After appyling feature engineering, all models performed better with lower RMSLE scores. Random Forest and XGBoost seem to overfit since the RMSLE scores on the test set are way greater than those on the training set.
![](https://github.com/nhh979/Bike_Sharing_Demand_proj/blob/main/images/baseline_models_comparison.jpg)

- In the hyperparameter tuning phase, We found out that Lasso and Ridge models produced lower RMSLE score (about 10% difference) and less overfitting but still greatly different (about 2-3 times greater) in comparison with Random Forest and XGBoost models. Although Random Forest and XGBoost models with hyperparameter tuning outperformed Lasso and Ridge, their RMSLE score were a little larger (about 10-15%) compared with their baseline model. Besides, there is a bit gap between training and test RMSLE score for Random Forest, indicating  high overfitting. Meanwhile, XGBoost performance were better than Random Forest and less overfitting.
![](https://github.com/nhh979/Bike_Sharing_Demand_proj/blob/main/images/tuned_models.jpg)

- Results on Kaggle submissions: XGBoost model provided public and private score 0.47501, while this number is 0.59508 for Random Forest model. With the ratio XGBoost:RandomForest = 0.75:0.25 combination, the score is 0.47496 on both public and private leaderboard.
  
|Model|Public RMSLE|Private RMSLE|
|------|------|------|
|XGBoost|0.47501|0.47501|
|Random Forest|0.59508|0.59508|
|XGBoost & RandomForest|0.47496|0.47496|

## VI. Conclusion
In summary, this notebook conducted a comprehensive analysis of hourly bike rental data spanning two years. It encompassed data exploration, preprocessing, and feature engineering to prepare the data for modeling. The exploratory data analysis provided valuable insights into rental patterns based on different factors, such as weather, day of the week, and hour of the day.

The model selection process involved evaluating three regression algorithms, ultimately identifying the Random Forest model as the best performer based on the Root Mean Squared Logarithmic Error (RMSLE) metric. This model was then utilized to predict bike rental counts on the test data, and the results were submitted for evaluation.

Notably, the analysis unveiled intriguing patterns, showcasing the distinct rental behaviors between registered users and casual users. The findings indicated that registered users primarily used bikes for their daily work commute, whereas casual users showed higher rentals during weekends, likely for recreational purposes.

However, future improvements could involve exploring additional features, employing advanced feature selection techniques, and experimenting with various regression algorithms to further enhance the model's accuracy and generalizability. The analysis and insights presented here can provide valuable guidance for bike-sharing companies to optimize their services and meet the diverse preferences of their user base.
