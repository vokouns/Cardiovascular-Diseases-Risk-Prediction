# Cardiovascular-Diseases-Risk-Prediction

## Project Overview
Our group decided to focus our project on heart disease or cardiovascular disease (CVD) and potential health indicators that may predict the likelihood of an individual being at risk for developing CVD.


## Data Analysis
We found our health indicator data via Kaggle, which is from the 2021 BRFSS Dataset from the CDC. The data was pretty clean, so we didn't have to make too many changes. We did remove the data under the checkup column. 

Once that data was removed, we converted it to be inputted into our modules. First, we scaled all of the numeric data using StandardScaler. Second, we transformed all of the categorical data using get_dummies. Once those changes were made, we concatenated our new data frames into one transformed data frame. We then used this data for our module evaluations.

We first began with a deep neural network model. We utilized three layers and various hidden node values. Upon completing the model, we got an accuracy score of 91%. With further evaluation using the confusion matrix and classification reporting, we found that while the accuracy score was still in the low 90%s, the precision of detecting heart disease was really low at 37%. We decided to try out an additional module. We then implemented a Random Forest model, which garnered very similar results. 

We then began further optimization by tweaking the two previous models and adding a logistical regression model. Initial optimization showed that the logistical model had the best accuracy score and best scores when looking at the confusion matrix and classification reporting.

We moved forward with this model as our prediction model and downloaded it into our code. We decided to create an online platform where individuals could input their health data to determine whether they were at risk for CDV. We created our Flask and Python files to run the site.

## References
Kaggle - https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset

Utilized module challenge homework and weekly exercises for module coding examples and references.