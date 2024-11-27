# Cardiovascular Diseases Risk Prediction

## Project Overview
Our group decided to focus our project on heart disease or cardiovascular disease (CVD) and potential health indicators that may predict the likelihood of an individual being at risk for developing CVD. 

We found our health indicator data via Kaggle, which is from the CDC's 2021 BRFSS Dataset. The data was clean, so we didn't have to make too many changes. However, we removed the data under the checkup column prior to analysis.

We also looked at all of our data to evaluate any trends by uploading our original data file into Tableau to create a variety of charts and graphs. We created charts for:

    Total of people with heart disease.
    Total patients by sex.
    People with heart disease by sex.
    Overall health categories.
    All categories versus heart disease.

## Data Analysis
Since our data was pretty clean and only minor changes needed to be made, we decided to run multiple analysis models.

Prior to running our models, we plotted a correlation matrix for all of our data to see if it would tell us anything. Due to the amount of data and various categories, the chart doesn't highlight any starting information as it's related to heart disease.

To start the analysis, we had to convert the data first. First, we scaled all of the numeric data using StandardScaler. Second, we transformed all of the categorical data using get_dummies. Once we made those changes, we concatenated our new data frames into one transformed data frame, which we then used for our module evaluations.  

We began with a deep neural network model. We imported our dependencies, split our data into features and arrays, created the StandardScaler, fitted the scaler, scaled the data, and then defined and trained the model. We leveraged a confusion matrix and classification report to see how our model performed. In the initial model, the accuracy was 92%, but the precision of heart disease was 33%. 

In the initial analysis, we also used a Random Forest model to see if it would produce better results. The Random Forest model had an accuracy score of 92% and a precision score of 40% for heart disease. 

In our optimization, we decided to utilize fewer layers (three), various hidden node values, and different random states. Upon completing the neural network model, we got an accuracy score of 91% and a precision of around 40%. Results remained very similar. Upon further inspection of our data, we found that the data we had didn't have enough people in the data set of individuals with heart disease in order for the model to predict heart disease accurately, so we needed to change the amount of data in training and testing. We changed the split ratio of training and testing data and reran everything. With further evaluation of these changes using the confusion matrix and classification reporting, we found that while the overall accuracy score was still in the low 90%s, the precision of detecting heart disease was higher than our initial model but still pretty low at 49%. 

We decided to try two additional modules with similar optimizations to see if we could increase the precision score. First, we ran the Random Forest model again with the new split, which garnered very similar results to the first try. The Random Forest model had an accuracy score of 92% for heart disease but a precision score of 41%. 

We decided to add a linear regression model to our optimization. We split and trained the data similarly to previous models. We also created a confusion matrix and classification report. For heart disease, this model's results were 92% accuracy and 55% precision, which were still low but higher than the other two models.

We moved forward with the linear regression model as our prediction model since these were the highest results. We ran a quick test to see if our model would make a prediction, which it did. We then decided to create an online platform where individuals could input their health data to determine whether they were at risk for CDV. 

## Site Creation
To build our site, we had to create our Flask and Python files to run it. The index file holds all of our coding for how the site looks, including headers, boxes for our input data, titles, an about section, and a contact section. 
The result file contains the coding for our results page that populates once someone has inputted their information into the model. Our Python files contain all of our code that takes in someone's data entries, evaluates them, and makes the prediction to determine if they are at risk for CVD or not.

## References
Kaggle - https://www.kaggle.com/datasets/alphiree/cardiovascular-diseases-risk-prediction-dataset
Gemini - leveraged Colab's Gemini tool to troubleshoot coding errors while running our models and creating our charts.
Alamy - reference for our BMI index chart for the website
Utilized module challenge homework and weekly exercises for module coding examples and references.