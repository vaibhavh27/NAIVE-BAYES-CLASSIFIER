# NewsPalette

## Introduction
This project focuses on classifying news articles into different categories such as wellness, politics, entertainment, travel, style, parenting, food, world news, business and sports. The classification is done using the Naive Bayes algorithm. The dataset used for training and testing was sourced from Kaggle.

## Dataset Preprocessing
Before applying the Naive Bayes algorithm, several preprocessing steps were carried out to clean and prepare the dataset:

### 1. Removing Null Values
Rows containing null values were removed from the dataset to ensure data integrity.

### 2. Removing Duplicates
Duplicate entries in the dataset were removed to avoid bias and redundancy in the training process.

### 3. Removing Unwanted Columns
Columns that were not relevant to the classification task were removed from the dataset to reduce dimensionality.

### Text Processing
The text data in the dataset underwent several preprocessing steps to standardize the format and make it suitable for analysis:

#### i) Removing Punctuation
Punctuation marks such as periods, commas, and exclamation points were removed from the text data as they do not typically contribute to the classification task.

#### ii) Converting to Lower Case
All text data was converted to lowercase to ensure consistency in word representations and avoid duplication of words due to case sensitivity.

#### iii) Removing Digits
Numeric digits were removed from the text data as they are not typically indicative of news article categories.

#### iv) Removing Stopwords
Stopwords, which are common words like 'and', 'the', and 'is', were removed from the text data as they do not carry significant meaning for the classification task.

#### v) Lemmatization
Lemmatization was performed to reduce words to their base or root form, which helps in standardizing the vocabulary and improving classification accuracy.

## TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) vectorization was applied to convert the text data into numerical features. This technique represents each document as a vector of word frequencies weighted by their importance in the corpus. It helps in capturing the significance of words in distinguishing between different news categories.

## Naive Bayes Classification
The Naive Bayes algorithm was chosen for this classification task due to its effectiveness in handling text data. Naive Bayes assumes that features are conditionally independent given the class, which makes it particularly suitable for text classification tasks where the presence of one word may not necessarily affect the presence of another.

## News Categories
The dataset includes news articles categorized into the following classes:
- Wellness
- Politics
- Entertainment
- Travel
- Style
- Parenting
- Food
- World News
- Business
- Sports

## Front End Development with Streamlit
A front-end interface was developed using Streamlit to interact with the Naive Bayes classification model. Streamlit provides a user-friendly way to build interactive web applications for machine learning and data science projects.

##Image Gallery
![Alt Text](/assets/Img-1.png?raw=true)
![Alt Text](https://github.com/ParthaSarathi-23/Newsification/blob/main/Img-2.png?raw=true)
![Alt Text](https://github.com/ParthaSarathi-23/Newsification/blob/main/Img-3.png?raw=true)
![Alt Text](https://github.com/ParthaSarathi-23/Newsification/blob/main/Img-4.png?raw=true)
