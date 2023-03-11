import sys
import re
import pickle
import pandas as pd
import numpy as np
from sqlalchemy import create_engine
import nltk
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from nltk.corpus import stopwords
nltk.download(["wordnet", "punkt", "stopwords"])

from sklearn.pipeline import Pipeline
from sklearn.feature_extraction.text import CountVectorizer, TfidfTransformer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.ensemble import AdaBoostClassifier
from sklearn.multioutput import MultiOutputClassifier
from sklearn.metrics import confusion_matrix, classification_report

def load_data(database_filepath):

    """
    Function:
    Load dataset from database
    
    Args:
    database_filepath(str): the file path of the database
    
    Return:
    X(DataFrame): messages feature dataframe
    Y(DataFrame): target dataframe
    """
    
    engine = create_engine("sqlite:///{}".format(database_filepath))
    df = pd.read_sql_table("disaster_messages_table", engine)
    X = df.messages
    Y = df.iloc[:, 4:]
    return X, Y
    

def tokenize(text):

    """
    Function:
    Split sentences into words then return the root form of the words (or lemmatized words)
    
    Args:
    text(str): the input message
    
    Return:
    lemmed(list): a list of lemmatized words
    """
    
    # Normalize text
    text = re.sub(r"[^a-zA-Z0-9]", " ", text.lower())
    # Tokenize text
    words = word_tokenize(text)
    # Remove stop words
    words = [w for w in words if w not in stopwords.words("english")]
    # Lemmatize words
    lemmed = [WordNetLemmatizer().lemmatize(w) for w in words]
    
    return lemmed
    

def build_model():

    """
    Function:
    Build a model to classify disaster messages
    
    Return:
    cv(list): classification model
    """
    
    # Create a pipeline
    pipeline = Pipeline([
        ('vect', CountVectorizer(tokenizer=tokenize)),
        ('tfidf', TfidfTransformer()),
        ('clf', MultiOutputClassifier(AdaBoostClassifier()))
    ])
    
    # Create Grid search parameters
    parameters = {
        'tfidf__use_idf': (True, False),
        'clf__estimator__n_estimators': [50, 70, 80]
    }
    
    cv = GridSearchCV(pipeline, param_grid=parameters)
    
    return cv


def evaluate_model(model, X_test, Y_test, category_names):
    
    """
    Function:
    Evaluate input model then output classification report for each category used for model prediction
    
    Args:
    model: the classification model
    x_test: test messages
    Y_test: test targets
    category_names: names of categories
    """
    
    y_pred = model.predict(x_test)
    
    for i in range(len(Y_test.columns)):
        print(Y_test.columns[i], ":")
        print(classification_report(Y_test[Y_test.columns[i]], y_pred[:, i]))
    
    accuracy = (y_pred == Y_test.values).mean()
    print("The model accuracy: {:.3f}".format(accuracy))


def save_model(model, model_filepath):
    
    """
    Function:
    Save pickle file for input model
    
    Args:
    model: the classification model
    model_filepath(str): the output path of the pickle file
    """
    
    with open (model_filepath + "classifier.pkl", "wb") as f:
        pickle.dump(model, f)


def main():
    if len(sys.argv) == 3:
        database_filepath, model_filepath = sys.argv[1:]
        print('Loading data...\n    DATABASE: {}'.format(database_filepath))
        X, Y, category_names = load_data(database_filepath)
        X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.2)
        
        print('Building model...')
        model = build_model()
        
        print('Training model...')
        model.fit(X_train, Y_train)
        
        print('Evaluating model...')
        evaluate_model(model, X_test, Y_test, category_names)

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