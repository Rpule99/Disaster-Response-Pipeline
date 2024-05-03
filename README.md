<h1 align="center"> Disaster-Response-Pipeline/h1>

<div align="center" >
  <img src="https://img.shields.io/badge/made%20by-Rethabile%20Pule-blue?style=for-the-badge&labelColor=20232a" />
  <img src="https://img.shields.io/badge/Python 3.12.2-20232a?style=for-the-badge&logo=python&labelColor=20232a" />
</div>

## Table of Contents
* [Project Description](#ProjectDescription)
* [Project Libraries](#ProjectLibraries)
* [Project Files](#ProjectFiles)
* [Running the project](#RunningTheProject)
* [About the Model](#about-the-model)

## Project Description
This project aims to classify messages sent during a disasters, using NLP to classify each message into 36 different categories which include the following categories:
- genre
- related
- request
- offer
- aid_related
- medical_help
- medical_products
- search_and_rescue
- security
- military
- child_alone
- water
- food
- shelter
- clothing
- money
- missing_people
- refugees
- death
- other_aid
- infrastructure_related
- transport
- buildings
- electricity
- tools
- hospitals
- shops
- aid_centers
- other_infrastructure
- weather_related
- floods
- storm
- fire
- earthquake
- cold
- other_weather
- direct_report

Classifying these messages will enable us to determine the current situation experience during the disaster. This will enable 1st responders and emergency departments to prepare themselves to deal with the situation. For examples if most of the messages in an area experiencing natuaral disaster such as a flood, and there are multiple injuries or death, emergency responders can emobilise themselves to better deal with the situation. The classifications can also enable better coverage of the disaster to the general public as we have multiple expirences from the victims.

The Project will be the starting point of enabling this where we begin by cleaning the data and saving the clean data in a table in a database. Once the data is saved the trained classifier will select the data and train a classifier model using grid search for optimal hyperparameters, the model will then be saved. Lastly, a web app will be used to classify new messages and show case graphs of the training data.

## Project Libraries

To run this project the following packages and libraries are required.
- SQLalchemy: to query the database and save data.
- pandas: this will assist in manipulating the data and training model.
- data_wrangling, data_clean, data_saver: these are modules created for this project and can be found in the project files.
- nltk: Used to tokenize and process the text data, in order for the model to train on.
- sklearn: ML package used for training and improving the model.
- xgboost: ML package which includes the classifier used to train the model.
- pickle: In order to save the final model.
- flask: used to create the web app.
- plotly: package used to aid in rendering interactive graphs.

## Project Files

- Root Directory:
    - App
        - templates
            - go.html: Renders the model output
            - master.html: index of webapp, renders graphs
        - app.py: houses flask app, and initialises web app
    - Data
        - categories.csv: Raw catagory data
        - messages.csv: Raw messages data
        - data_clean.py: module to remove duplicates
        - data_wrangling.py: module to wrangle and clean data
        - Disaster.db: SQLlite DB to save data
        - process_data.py: executed file to clean and save data
    - Models
        - data_saver.py: module to save data into a the db 
        - new_best_model.pkl: Pickle file which stars the saved model
        - train_classifier: trains and saves the classifier model


## Running the Project

The project can be run via the terminal, just ensure to install the necessary packages in your environment. To run the project follow the steps


1. Run the command to run the ETL pipeline and save the clean data and save into the DB.
```
python3 data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.DB
```
1. Run the command to run the classifier trainer and save the model. This will also save the classifier metrics.
```
models/train_classifier.py data/DisasterResponse.DB models/new_best_model.plk
```
1. lastly run to view the web app and start classifying messages. The terminal will display ip where you can click and the web app will be shown in the default browser.
```
'app/app.py' 
```

## About the model

The model obtained in the project had an overall accuracy of 0.95 on the test, which is accurate. However, we also discovered that the training data was very imbalanced which reduces our ability to trust this metric. We also had moderate performance with the recall (0.55). Where we have average instances of true positives. To improve this model we could employ SMOTE where we oversample the least occurring cases, meaning if one category is observed at 1% in the data we can create more instances of this case, alternatively reduce instances of the majority cases.
