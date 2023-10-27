# Using Machine Learning to Predict Substance Use Treatment Success
![Predicting a Brighter Future A Data-Driven Approach to Substance Treatment Success](https://github.com/chicofanito/Capstone-2/assets/59300889/5d7a3767-73ec-4a5c-a879-1bb6f292ab1a)

## Introduction
According to the Centers for Disease Control
and Prevention (CDC), more than one million
people have died since 1999 from a drug
overdose. In 2021, 106,699 drug overdose
deaths occurred in the United States. The
age-adjusted rate of overdose deaths
increased by 14% from 2020 (28.3 per
100,000) to 2021 (32.4 per 100,000).This project aims to leverage machine learning
techniques to investigate disparities in
substance use disorder (SUD) treatment
success. The primary objective is to develop
predictive models capable of identifying
individuals who are likely to successfully
complete substance use treatment programs.
The project's investigation is motivated by the
disparities in SUD treatment services and
outcomes. By identifying key predictors of
successful treatment and revealing disparities,
this project seeks to illuminate strengths and
weaknesses in service delivery. This analysis has
the potential to boost treatment success rates,
ultimately addressing unmet treatment needs.

## 1. Data
The project utilized data from the Treatment Episode
Data Set: Discharge (TEDS-D), sourced from the
Substance Abuse and Mental Health Services
Administration (SAMHSA). Data regarding the number of treatment facilities and response rates was extracted from SAMHSA's PDF reports. Additionally, population statistics and geographical details were sourced from Wikipedia. Links to these information are listed below:

* [TEDS-D DATA](https://www.datafiles.samhsa.gov/dataset/treatment-episode-data-set-admissions-2020-teds-2020-ds0001)
* [Facility Data](https://www.samhsa.gov/data/sites/default/files/reports/rpt35969/2020%20NSSATS%20State%20Profiles_FINAL.pdf)
* [Wikipedia Population Data](https://simple.wikipedia.org/w/index.php?title=List_of_U.S._states&oldid=7168473)

## 2. Data Wrangling
[Full Data Wrangling Report](https://github.com/chicofanito/Capstone-2/blob/6467d9de1632185f90dbf203b09615695727ca3d/notebooks/Capstone%202%20-%20Substance%20Use%20Treatment%20-%20Data%20Wrangling.ipynb)

In the process of preparing the TEDS-D dataset for analysis, I had a few data-wrangling challenges. An overview of the main issues is presented below:

**Problem 1: The TEDS-D included all admissions and discharges, rather than individual cases.**
>**Solution: To tackle this, a filtering approach was applied to retain only those records corresponding to individuals without prior substance use disorder (SUD) treatment. This transformation aligned the dataset with the goal of studying unique individual instances and resulted in a dataset containing 503,107 distinct individuals.**

**Problem 2: Significant amount of missing data**
>**Solution: While this might not be ideal as some important variables may have been lost, I excluded records with missing values in any of the predictors and outcome variables. This ensured the integrity of the data used for analysis.**

**Problem 3 No single source providing information on the total treatment facility count by state.**
>**Solution: To overcome this challenge, data was extracted from PDF reports to acquire details on the number of treatment facilities surveyed by The Substance Abuse and Mental Health Services Administration and response rates. By combining response rates with the count of facilities surveyed, the total number of treatment facilities for each state was accurately computed.**

**Problem 4: Misspelled state names and special characters within the dataset**
>**Solution: To facilitate data merging by state, all special characters were removed, and state name spellings were corrected.**

## 3.Exploratory Data Analysis

[Full EDA Report](https://github.com/chicofanito/Capstone-2/blob/076e87994a52d55b0afb9871d50bf36c89e0c72c/data/Capstone%202%20-%20Substance%20Use%20Treatment%20-%20EDA.ipynb)

The data indicates an improvement in completion rates beyond the age of 44, with the 12-14 age group displaying the lowest completion rate.

![image](https://github.com/chicofanito/Capstone-2/assets/59300889/064818bf-6c9a-4255-b8cc-3fafb3d7ba17)


Boxplot analysis suggests that there could be disparities in the distribution of treatment facilities per area between cases of completed and incomplete treatment.

![image](https://github.com/chicofanito/Capstone-2/assets/59300889/18ccca22-9a6a-4640-bd46-4da24ec39378)


## 4. Preprocessing
[Preprocessing Report](https://github.com/chicofanito/Capstone-2/blob/076e87994a52d55b0afb9871d50bf36c89e0c72c/data/Capstone%202%20-%20Substance%20Use%20Treatment%20-%20Preprocessing.ipynb)

In the data preprocessing phase, several essential steps were taken to prepare the dataset for machine learning modeling. These steps included calculating the population per square mile, selecting relevant columns, creating dummy variables for categorical features, standardizing numeric attributes, and encoding the target variable.

## 5. Modelling & Evaluation
[Modelling Report](https://github.com/chicofanito/Capstone-2/blob/076e87994a52d55b0afb9871d50bf36c89e0c72c/data/Capstone%202%20-%20Substance%20Use%20Treatment%20-%20Modeling.ipynb)

This is a classification problem as we are trying to predict treatment outcomes (complete or incomplete). I explored these
conventional machine learning models to build the model and compare performances:
* Logistic Regression:
* Random Forest:
* K-Nearest Neighbor (KNN)
* Naive Bayes
* Gradient Boosting

The Random Forest and Gradient Boosting models outperformed the other models. I choose to pursue the Random Forest Tree as my final algorithm. 

![image](https://github.com/chicofanito/Capstone-2/assets/59300889/2dcbd400-007c-4d5a-956e-53183e80c801)


## 6. Hyperparameter Tuning Random Forest Model

I utilized RandomizedSearchCV to tune the model with a randomized search over hyperparameters to reduce the computational cost compared to using grid search. 
While accuracy increased slightly, ROC-AUC Score decreased after hyper tuning using the random search. The decrease in the ROC-AUC score after hyperparameter tuning using random search could be due to the randomness involved in the search process. Randomized search explores a random subset of the hyperparameter space, and in some cases, it may not find hyperparameters that improve the model's performance on your specific dataset.
Hence, I performed cross-validation to get a more stable estimate of the model's performance. This helped ensure that the hyperparameters were chosen based on more robust performance estimates.
I performed 5-fold cross-validation on your Random Forest model, and the results indicate that the mean ROC-AUC score of approximately 0.879 suggests that your Random Forest model is performing well on the cross-validated subsets of your training data. The standard deviation is relatively small, indicating that the model's performance is consistent across different folds. This information provides more confidence in the model's predictive ability and its stability when applied to unseen data.

## 7. Top 10 Features

The features that were most influential in predicting treatment success were:

1.	**The Census division**, specifically "Mountain," appears to be highly influential. It suggests that treatment outcomes may vary significantly by region, with the Mountain division having a strong positive impact on treatment success.
2.	**Population density per square mile** is the second most important feature. It indicates that areas with higher population density may have a positive influence on treatment success. This could be due to better access to resources and support services.
3.	**The state of Arizona** seems to be a crucial predictor, suggesting that individuals receiving treatment in Arizona may have a higher likelihood of successful treatment completion.
4.	**The total number of treatment facilities** in the state is a significant factor. More treatment facilities could mean better access to care, increasing the chances of successful treatment outcomes.
5.	**Individuals who do not report alcohol at admission** have a positive impact on treatment success. This feature suggests that those without alcohol use issues have a higher likelihood of completing treatment successfully.
6.	Conversely, individuals who report using only other drugs (not alcohol) at admission also contribute positively to treatment success.
7.	**Kentucky's state-specific impact** on treatment outcomes is relatively smaller than Arizona, but it still plays a role in predicting success.
8.	Longer lengths of stay within the range of **61-90 days** have a positive influence on treatment success. This suggests that extended treatment durations may be more effective.
9.	Similarly, very long stays (**181-365 days**) also contribute positively to successful treatment outcomes.
10.	Stays within the range of **91-120 days** are another important factor, indicating that moderate-duration treatments have a favorable impact.

## 8. Final Predictions

My Random Forest model has achieved notable success in predicting treatment outcomes, as indicated by an accuracy of
approximately 80.56%. This means that roughly 80.56% of the cases were correctly classified by the model. Additionally,
the precision score of about 81.30% demonstrates that, among the cases predicted as successful treatment completions,
the majority were indeed accurate predictions.

However, there are some downsides to consider. The model's recall, at around 63.70%, suggests that it missed identifying a
substantial portion of actual successful treatment completions. This means there is room for improvement in capturing
more positive cases. The F1-Score, which balances precision and recall, stands at about 71.43%, indicating a reasonably
good balance, but further optimization may enhance overall performance. Lastly, the ROC-AUC score, approximately
77.33%, suggests the model's ability to distinguish between the two classes is moderately effective, but fine-tuning could
boost this aspect of the model.

<img width="446" alt="image" src="https://github.com/patricotyrell/Capstone-2/assets/59300889/a40a96f9-4e2e-4fc1-92b1-da3a53a80d7b">

<img width="445" alt="image" src="https://github.com/patricotyrell/Capstone-2/assets/59300889/643f4077-a9b5-4e09-b9cb-2c80300e7f14">


## 9. Future Directions

While my journey through model selection and hyperparameter tuning has yielded encouraging results, there are several avenues for future exploration and improvement:

1. **Feature Engineering**: Continue to explore feature engineering techniques to create new variables or transform existing ones, potentially uncovering additional patterns that can enhance predictive performance.
   
2. **Feature Importance**: Dig deeper into the feature importances generated by the Random Forest model.

3. **Add more Variables**: Explore the potential impact of socioeconomic and demographic factors on treatment success, such as income, education, race, ethnicity, sex, and employment status.

4. **Ensemble Methods**: Experiment with other ensemble methods like XGBoost, LightGBM, or AdaBoost to determine if any of them can surpass the performance of Random Forest.

5. **Imbalanced Data Handling**: Since the dataset is highly imbalanced (more incomplete cases than complete cases), I can explore techniques such as oversampling, undersampling, or the use of synthetic data generation methods to mitigate class imbalance.

6. **Model Interpretability**: Investigate techniques for improving the interpretability of Random Forest models, as they can sometimes be seen as "black boxes." Techniques like SHAP (SHapley Additive exPlanations) values or partial dependence plots can shed light on model predictions.

7. **Deployment**: Since I would like to use the model in practice, I will have to consider the deployment pipeline, model monitoring, and integration with existing systems.
Ethical Considerations: I have to ensure that the model is used in an ethical and responsible manner, avoiding biases and unintended consequences.


## 10. Credits

I would like to express my gratitude to the Schenectady Public Health Services for generously funding my course. Special thanks go to my Data Science mentor, Upom Malik, for his invaluable guidance and advice.



