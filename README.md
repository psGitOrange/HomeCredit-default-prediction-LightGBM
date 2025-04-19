# HomeCredit-default-prediction-LightGBM
This machine learning project focuses on the "Home Credit Default Risk" dataset, a Kaggle competition hosted by Home Credit to predict applicant's ability to repay Credit/Loan.

###Competition
The Competition was hosted on **kaggle** on **May,2018**. running for more than 3 months with a deadline for final submission on **Aug,2018**

**Description**</br>
Home Credit strives to broaden financial inclusion for the unbanked population by providing a positive and safe borrowing experience. In order to make sure this underserved population has a positive loan experience,
Home Credit makes use of a variety of alternative data--including telco and transactional information--to predict their clients' repayment abilities.
They employ diverse statistical and machine learning techniques for these predictions. By putting up the competition, they're challenging kagglers to help them unlock the full potential of their data.

Well the competition is already over, we can still do late submissions to get the evaluations done on our predictions to give score

**Evaluation**</br>
Submissions are evaluated on area under the ROC curve between the predicted probability and the observed target.

**Submission File**</br>
For each SK_ID_CURR in the test set, you must predict a probability for the TARGET variable. 

**Data**
![data relationship diagram](https://storage.googleapis.com/kaggle-media/competitions/home-credit/home_credit.png)

**application_{train|test}.csv**</br>
- This is the main table, broken into two files for Train (with TARGET) and Test (without TARGET).
- Static data for all applications. One row represents one loan in our data sample.

**bureau.csv**</br>
- All client's previous credits provided by other financial institutions that were reported to Credit Bureau (for clients who have a loan in our sample).
- For every loan in our sample, there are as many rows as number of credits the client had in Credit Bureau before the application date.

**bureau_balance.csv**</br>
- Monthly balances of previous credits in Credit Bureau.
- This table has one row for each month of history of every previous credit reported to Credit Bureau – i.e the table has (#loans in sample * # of relative previous credits * # of months where we have some history observable for the previous credits) rows.

**POS_CASH_balance.csv**</br>
- Monthly balance snapshots of previous POS (point of sales) and cash loans that the applicant had with Home Credit.
- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credits * # of months in which we have some history observable for the previous credits) rows.

**credit_card_balance.csv**</br>
- Monthly balance snapshots of previous credit cards that the applicant has with Home Credit.
- This table has one row for each month of history of every previous credit in Home Credit (consumer credit and cash loans) related to loans in our sample – i.e. the table has (#loans in sample * # of relative previous credit cards * # of months where we have some history observable for the previous credit card) rows.

**previous_application.csv**</br>
- All previous applications for Home Credit loans of clients who have loans in our sample.
- There is one row for each previous application related to loans in our data sample.

**installments_payments.csv**</br>
- Repayment history for the previously disbursed credits in Home Credit related to the loans in our sample.
- There is a) one row for every payment that was made plus b) one row each for missed payment.
- One row is equivalent to one payment of one installment OR one installment corresponding to one payment of one previous Home Credit credit related to loans in our sample.

**HomeCredit_columns_description.csv**</br>
- This file contains descriptions for the columns in the various data files.

##Summary
- We explored our application training data performed label and one hot encoding for categorical data and Scaling and imputing for numerical data.</br> Removed missing values and correlated features, feature engineering to get polynomial and domain specific features.</br> Performed baseline prediction using Random forest and lightgbm model.
- Following the steps we added Bureau and Bureau balance data to application data, Bureau data was  aggregated w.r.t application id where it had multiple bureau Id  for previous credits.</br> **KDE** plots to showing correlation of feature between defaulter and non defaulter applicants. Removed  missing value and correlated features.</br>
Integrated with bureau balance data aggregating monthly balance for each bureau Id.</br>
- Used application and bureau features for prediction with **LightGBM** model, obtaining roc-auc score around 0.75, i.e. predicting default risk with 75% accuracy.
-Previous application had previous loan data at home credit,
 aggregated to get features related to application.
Pos cash balance had monthly loan details, credit balance had monthly balance details, installment and monthly payment details resp., we aggregate it w.r.t to previous application id then with application id.
- **Collated** all application-bureau data and application-previous data to obatain final dataset. Then performed feature selection for last time by correlated,  missing features and zero importance features got from model.
- Performed predictions on final dataset and got 0.7855 score.</br> Also removed 5% features with least importance to get smaller subset of features, we got **0.78622** roc-auc score.
- For hyperparameter optimisation we first selected small dataset and found out n_estimators by early stopping. Used this with other found hyper parameters to get improved roc-auc score.</br>
But this optimised hyperparameters didn't do well on full data set.

##Inferences
1. Found out **anomaly** in days employed and replaced them with np.nan, as all anomalous rows contained same value. And created another column to tell the machine learning model that the certain feature was anomalous.
2. KDE plots for correlating feature with target varaible gave **variance**, which showed its importance for model's performance.
3. **Feature engineering**:</br>
 With label encoding categorical features, aggregating numeric features,adding polynomial features and Domain relevent features.</br>
 Then by adding more features from related files, removing correlated and missing value features this models performance increased from 0.0.69252 to 0.78622 roc-auc score.
4. Created more than **1500 features** in the process, rigourously reducing it to 483 features for our final dataset, further reduced to **313 features** by remove least 5% import features.
5. Improved model performance by add more related data provided by Home Credit, about **6 files**. From 0.75336 score on application data to 0.76031 with bureau data and jumping to **0.78622 roc-auc score** with add previous application data.
6. Performed **Hyperparameter Optimization** with **RandomSearchCV** to calculate hyper parameters on a small data set and then we applied it to A final data set but found out that this parameters don't translate on the full data and we got relatively less score on our predictions
7. Coverting to lower datatypes, storing and retrieving data in parquet and feather files boosted I/O performance and reduced **memory** and disk space.

##Conclusion
We executed a comprehensive **end-to-end** workflow including meticulous data preprocessing, analysis features, extensive feature engineering, and the development of machine learning **default risk prediction** models.

For all our datasets we followed the similar **steps**:
- **Analyse** few important features with bar graphs, plotted kde graphs to view feature varaince between defaulter and non-defaulters.
- **Preprocess** data i.e. convert datatypes, label encoding, scaling and imputing.
- **Feature Engineer** by adding polymial, domain specific and aggregated features to dataset.
- Reviewed **correlated** and **Missing value** features, and removed features with high correlation and missing value count.
- Data through ML **model** for metric evaluation and scraping out features with zero importance.

**LightGBM** (Gradient Boosting Model) model significantly improved performace as compared to Randomforest. As GBMs try to reduce error/ loss itself by building trees on top of other, Ultimately imporveing performance.

We used '**RandmonSearchCv**' for hyperparameter optimzation on sample dataset, we had better score than model without optimization.

We got **0.78622 roc-auc-score** on competitions private test dataset, which was less by 0.02 than than the highest scoring model. So we can conclude overall we have build a solid ML model for default risk prediction.

##Future Work

- Theres much room for Exploratory Data Analysis(**EDA**) as there are many features in training data related files.In our final dataset if we had used only available features we could analyse each applicants credit, balance, intallments, etc.
- Creating more features using **automated feature engineering** tools and then taking out valueable features or by selectively creating features with domain knowledge a better generalizing machine learning model can be made.

- Hyperparameter optimization through state of art automated tools like **bayesian optimization.**

- Improving model performance by **stacking** different models or by using Deep learning model.

## References

Data Manupilation Libraries
- Pandas documentation: https://pandas.pydata.org/docs/user_guide/index.html
- Ultimate Guide to Handling Missing Data: https://soumenatta.medium.com/the-ultimate-guide-to-handling-missing-data-in-python-pandas-a6b0913a7cd4

EDA Libraries
- Matplotlib documentation: https://matplotlib.org/stable/index.html
- seaborn homepage: https://seaborn.pydata.org/

Machine learining Libraries
- scikit-learn library https://scikit-learn.org/stable/, for documentation on label encoder and onehot encoding and machine learning models like randomforest
- lightgbm documentation https://lightgbm.readthedocs.io/en/stable/

Kaggle Notebooks
- kaggle notebook for a complete guide https://www.kaggle.com/code/willkoehrsen/start-here-a-gentle-introduction
- Another kaggle notebook for feature engineering https://www.kaggle.com/code/jsaguiar/lightgbm-with-simple-features
