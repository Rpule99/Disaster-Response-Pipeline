# Disaster-Response-Pipeline

This project aims to classify messages sent during disasters, using NLP to classify each messages into 36 diffrent catagories which include the following catagories:
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

This will be done by Cleaning the data and saving the clean data in table in a database. Once the the data is saved the train classifer will select the data and train a classifer model using grid search for optimal hyper parameters, the model will then be saved. Lastly a web app will be used to classify new messages and show case graphs of the training data.

## Requirments

In order to run this project the following packages and libraries are required.
- sqlalchemy: to query the database and save data.
- pandas: this will assist in manipulating the data and training model.
- data_wrangling, data_clean, data_saver: these are modules created for this project and can be found in the project files.
- nltk: Used to tokenize and process the text data, inorder for the model to train on.
- sklearn: ML package used for training and improving the model.
- xgboost: ML package which includes the classifer used to train the model.
- pickle: Inorder to save the final model.
- flask: used to create the web app.
- plotly: package used to aid in rendering interactive graphs.

## Running the Project

The project can be ran via the terminal, just ensure to install the nessesary packages in your environment. To run the project follow the steps

1. Run the command 'data/process_data.py data/messages.csv data/categories.csv data/DisasterResponse.DB' to run the ETL pipeline and save the clean data and save into the DB.
1. Run command 'models/train_classifier.py data/DisasterResponse.DB models/new_best_model.plk' to run the classifer trainer and save model. this will also save the classifer metrics.
1. lastly run 'app/app.py' to view the webapp and start classifying messages. The terminal will display ip where you can click and the web app will be shown in default browser.

## About the model

The model obtained in the project had an overall accuracy of 0.95 on the test, which is accurate. However we also discovered that the training data was very imbalanced which reduces our ability to trust this metric. We also had moderate performance with the recall (0.55). Where we have average instances of true positives. To improve this model we could employ SMOTE where we oversample the least occuring cases, meaning if one catagory is observed 1% in the data we can create more instances of this case, alternativly reduce instances of the mojority cases.
