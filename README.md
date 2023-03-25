# Disaster Response Pipeline Project


### Table of Contents

1. [Installations](#installation)
2. [Project Motivation](#motivation)
3. [Procedure Descriptions](#procedures)
4. [File Structure](#structure)
5. [Instructions](#instructions)
6. [Licensing, Authors, Acknowledgements](#licensingetc)


## Installations <a name="installation"></a>
Python (version 3.8.8)

Python libraries needed for the analysis:

- pandas
- numpy
- reobjective
- sys
- scikit-learn
- sqlite3
- sqlalchemy
- nltk
- pickle
- Flask
- plotly
- seaborn
- matplotlib


## Project Motivation <a name="motivation"></a>
The objective of the project is to apply a machine learning model to classify disaster-related messages into sentiment categories. 
One message can possibly belong to one or multiple sentiment categories.
The data of the project was provided by an AI company formerly named [Figure Eight](https://appen.com/) (now Appen).  
In the web app, classification results/ caegories are obtained whenever user input a message.


## Procedure Descriptions <a name="procedures"></a>
The project is consisted of three components:
1. **ETL Pipeline:** "process_data.py" contains the codes in the ETL pipline which:

    - Loads the messages and categories datasets
    - Merges the two datasets
    - Cleans the data
    - Stores it in a SQLite database
	
2. **ML Pipeline:** 'train_classifier.py' contains the codes in the ML pipeline which:

    - Loads data from the SQLite database
    - Splits the dataset into training and test sets
    - Builds a text processing and machine learning pipeline
    - Trains and tunes a model using GridSearchCV
    - Outputs results on the test set
    - Exports the final model as a pickle file

3. **Flask Web App:** enables users to type in disaster-related messages to explore which sentiment categories have strong relevance with inputs.


## File Structure <a name="structure"></a>

	- README.md: project brief displayed on Github
	- ETL Pipeline Preparation.ipynb: contains ETL pipeline preparation code
	- ML Pipeline Preparation.ipynb: contains ML pipeline preparation code
	- workspace
		- \app
			- run.py: flask file to run the app
			- \templates
				- master.html: main page of the web app 
				- go.html: result web page
				- css: folder containing css styling scripts for webpage rendering
				- js: folder containing javascript scripts for webpage interactions
		- \data
			- disaster_categories.csv: categories dataset
			- disaster_messages.csv: messages dataset
			- DisasterResponse.db: disaster response database
			- process_data.py: ETL pipeline
		- \models
			- classifier.pkl: the classifier selected (the optimized model)
			- train_classifier.py: classification code/ ML pipeline


## Instructions <a name="instructions"></a>
1. Run the following commands in the project's root directory to set up your database and model.

    - To run ETL pipeline that cleans data and stores in database
        `python data/process_data.py data/disaster_messages.csv data/disaster_categories.csv data/DisasterResponse.db`
    - To run ML pipeline that trains classifier and saves
        `python models/train_classifier.py data/DisasterResponse.db models/classifier.pkl`

2. Run the following command in the app's directory to run your web app.
    `python run.py`

3. Go to http://0.0.0.0:3001/


## Licensing, Authors, Acknowledgements <a name="licensingetc"></a>
Credits are given to [Figure Eight](https://appen.com/) for the data and to Udacity for the education. 