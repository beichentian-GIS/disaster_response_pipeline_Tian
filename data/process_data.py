import sys
import pandas as pd
import numpy as np
import sqlite3
import sqlalchemy import create_engine

def load_data(messages_filepath, categories_filepath):

    """
    Function:
    Load two dataset named "messages" and "categories" then merge them to a new dataset 
    
    Args:
    messages_filepath(str): the file path of the messages.csv
    categories_filepath(str): the file path of the categories.csv
    
    Return:
    df(DataFrame): a dataframe that combines the attributes of the two input datasets
    """
    
    messages = pd.read_csv(messages_filepath)
    categories = pd.read_csv(categories_filepath)
    df = messages.merge(categories, on="id", how="inner")
    
    return df


def clean_data(df):
    
    """
    Function: 
    Wrangle the merged dataframe
    
    Args:
    df(DataFrame): the original dataframe of messages and categories that needs to be wrangled
    
    Return:
    df(DataFrame): the wrangled dataframe of messages and categories
    """
    
    # Split categories into separate category columns
    categories = df.categories.str.split(";", expand=True)
    row = categories.iloc[0]
    category_colnames = row.apply(lambda x: x[:-2]).values.tolist()
    categories.columns = category_colnames
    
    # Convert category values to just numbers 0 or 1
    for column in categories:
        # set each value to be the last character of the string
        categories[column] = categories[column].str[-1]
        # convert column from string to numeric
        categories[column] = categories[column].astype(int)
        
     # Replace categories column in df with new category columns
     df.drop(["categories"], axis=1, inplace=True)
     df = pd.concat([df, categories], axis=1)
     
     # Remove duplicates
     df.drop_duplicates(inplace=True)
     
    return df


def save_data(df, database_filename):

    """
    Function:
    Output the wrangled dataframe to a database
    
    Args:
    df(DataFrame): the wrangled dataframe
    database_filename(str): the name of the output database
    """
    
    engine = create_engine('sqlite:///{}.db'.format(database_filename))
    df.to_sql('disaster_messages_table', engine, if_exists="replace", index=False)


def main():
    if len(sys.argv) == 4:

        messages_filepath, categories_filepath, database_filepath = sys.argv[1:]

        print('Loading data...\n    MESSAGES: {}\n    CATEGORIES: {}'
              .format(messages_filepath, categories_filepath))
        df = load_data(messages_filepath, categories_filepath)

        print('Cleaning data...')
        df = clean_data(df)
        
        print('Saving data...\n    DATABASE: {}'.format(database_filepath))
        save_data(df, database_filepath)
        
        print('Cleaned data saved to database!')
    
    else:
        print('Please provide the filepaths of the messages and categories '\
              'datasets as the first and second argument respectively, as '\
              'well as the filepath of the database to save the cleaned data '\
              'to as the third argument. \n\nExample: python process_data.py '\
              'disaster_messages.csv disaster_categories.csv '\
              'DisasterResponse.db')


if __name__ == '__main__':
    main()