import sys
import re
import string
import pandas as pd
import numpy as np
from nltk.tokenize import word_tokenize
from nltk.corpus import stopwords
from nltk.stem import WordNetLemmatizer
from sqlalchemy import create_engine
import nltk
nltk.download(['punkt', 'wordnet'])
nltk.download('stopwords')

from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline
from xgboost import XGBClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.feature_extraction.text import CountVectorizer
from nltk.tokenize import word_tokenize
from sklearn.metrics import f1_score, precision_recall_fscore_support
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import classification_report,accuracy_score
import data_saver as ds
import pickle


def load_data(database_filepath):
    #read data in
    db_path = 'sqlite:///'+database_filepath
    engine = create_engine(db_path)
    df = pd.read_sql_table('Message', engine)
    #create dummy variables
    one_hot_encoded_genre = pd.get_dummies(df['genre'], prefix='genre')
    one_hot_encoded_genre = one_hot_encoded_genre.astype(int) #turn true, false to 0 and 1
    df_encoded = pd.concat([df, one_hot_encoded_genre], axis=1)
    # split data
    X = df_encoded['message']
    Y = df_encoded.drop(columns=['id','message','original', 'genre'], inplace=False)
    category_names = Y.columns.tolist()
    return X, Y, category_names


def tokenize(text):
    text_no_pun = text.translate(str.maketrans("","", string.punctuation))#remove punctuation
    stop_words = set(stopwords.words('english')) #get stop word
    tokens = word_tokenize(text_no_pun)
    removed_stop = [word for word in tokens if word.lower() not in stop_words] #remove stop words
    lemmatizer = WordNetLemmatizer()

    clean_tokens = []
    for tok in removed_stop:
        clean_tok = lemmatizer.lemmatize(tok).lower().strip()
        clean_tokens.append(clean_tok)
    
    return clean_tokens


def build_model():
    pipeline_xgb =Pipeline([
                ('vect', CountVectorizer(tokenizer=tokenize)),
                ('tfidf', TfidfTransformer(use_idf = False)),
                ('clf', MultiOutputClassifier(XGBClassifier(n_estimators = 200, max_depth= 5))) ])
        
    return pipeline_xgb


def evaluate_model(model, X_test, Y_test, category_names):
    #predict
    predictions = model.predict(X_test)
    #Initiate metrics lists
    labels = category_names
    precision_list = []
    recall_list = []
    f1_score_list = []
    accuracy_list = []

    # loop of Labels and compute metrics
    for idx, label in enumerate(labels):
        y_true = Y_test.iloc[:, idx]  # Extract true labels for the current label
        y_pred = predictions[:, idx]  # Extract predicted labels for the current label
        
        # Compute precision, recall, F1-score, and accuracy for the current label
        precision, recall, f1_score, _ = precision_recall_fscore_support(y_true, y_pred, average = 'weighted')
        accuracy = accuracy_score(y_true, y_pred)
        
        # Append metrics to initiated lists
        precision_list.append(precision)
        recall_list.append(recall)
        f1_score_list.append(f1_score)
        accuracy_list.append(accuracy)

    # Create a DataFrame with metrics for each label
    metrics_df = pd.DataFrame({
        'Label': labels,
        'Precision': precision_list,
        'Recall': recall_list,
        'F1-score': f1_score_list,
        'Accuracy': accuracy_list
    })

    # transform the Label column
    metrics_df.set_index('Label', inplace=True)
    # Transpose so table is wider
    transposed_df = metrics_df.transpose()
    return transposed_df    


def save_model(model, model_filepath):
    with open(model_filepath, 'wb') as f:
        pickle.dump(model, f)

def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2, random_state=66)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        Model_metrics = evaluate_model(model, X_test, Y_test, category_names)
        
        #save metrics to DB
        ds.save_data_to_db(Model_metrics, 'Model_Metrics')

        print('Saving model...\n    MODEL: {}'.format(model_filepath))
        save_model(model, model_filepath)

        print('Trained model saved!')

    else:
        print('Please provide the filepath of the disaster messages database '\
              'as the first argument and the filepath of the pickle file to '\
              'save the model to as the second argument. \n\nExample: python '\
              'train_classifier.py ../data/DisasterResponse.db classifier.pkl')


if __name__ == '__main__':
    main()