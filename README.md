
# NewsPalette: News Article Classification
October 2024

## Introduction
NewsPalette is a machine learning project designed to classify news articles into predefined categories. This application utilizes the Naive Bayes algorithm to perform the classification, trained on a dataset sourced from Kaggle. The primary goal is to accurately categorize news content, providing users with an organized way to consume information based on their interests.

## Features
*   **Automated News Classification:** Predicts the category of a news article based on its headline (and potentially description).
*   **Interactive User Interface:** A user-friendly web application built with Streamlit allows users to input news details for on-the-fly classification.
*   **Categorized News Display:** Users can browse pre-loaded news articles sorted into dedicated tabs for each category.
*   **Model Performance:** Displays the accuracy of the classification model.
*   **Custom Styling:** Enhanced visual appeal through custom CSS integrated within the Streamlit app.

## News Categories
The model classifies news articles into the following 10 categories:
*   Wellness
*   Politics
*   Entertainment
*   Travel
*   Style & Beauty
*   Parenting
*   Food & Drink
*   World News
*   Business
*   Sports

## Technologies Used
*   **Programming Language:** Python
*   **Machine Learning:** Scikit-learn (for Naive Bayes, TF-IDF Vectorizer, metrics)
*   **Data Handling:** Pandas
*   **Web Framework:** Streamlit
*   **Model Persistence:** Pickle
*   **Development Environment:** Standard Python environment (e.g., VS Code, Jupyter Notebooks for model development)

## Dataset & Preprocessing
The dataset for this project was sourced from Kaggle (`NewsCategorizer.csv`). Rigorous preprocessing was crucial to prepare the data for the Naive Bayes algorithm.

### Data Cleaning
1.  **Removing Null Values:** Rows containing any null values were removed to ensure data integrity and prevent errors during model training.
2.  **Removing Duplicates:** Duplicate news entries were identified and removed to avoid bias and redundancy in the training data.
3.  **Removing Unwanted Columns:** Columns not relevant to the classification task (e.g., author names, specific dates if not used as features) were dropped to reduce dimensionality and focus on textual content.

### Text Processing
The textual data (primarily headlines and short descriptions) underwent several normalization steps:
1.  **Removing Punctuation:** All punctuation marks (e.g., '.', ',', '!', '?') were removed as they generally do not contribute significantly to topic classification.
2.  **Converting to Lower Case:** All text was converted to lowercase to ensure consistency and treat words like "News" and "news" as identical.
3.  **Removing Digits:** Numeric digits were removed from the text as they are typically not indicative of news article categories.
4.  **Removing Stopwords:** Common English words (e.g., 'and', 'the', 'is', 'a') that offer little semantic value for classification were removed using a standard stopwords list.
5.  **Lemmatization:** Words were reduced to their base or dictionary form (lemma) (e.g., "running" to "run", "studies" to "study"). This helps in consolidating different forms of a word into a single feature, improving model accuracy.

## Methodology

### TF-IDF Vectorization
TF-IDF (Term Frequency-Inverse Document Frequency) was employed to convert the preprocessed text data into numerical feature vectors. This technique assigns a weight to each word in a document based on its frequency within that document and its inverse frequency across the entire corpus. Words that are frequent in a specific document but rare in others receive higher TF-IDF scores, making them more significant for classification. The trained TF-IDF vectorizer is saved as `vector.pkl`.

### Naive Bayes Classification
The Multinomial Naive Bayes algorithm was chosen for this text classification task. It's a probabilistic classifier based on Bayes' theorem, assuming conditional independence between features (words) given the class. Despite its simplicity, Naive Bayes is highly effective and computationally efficient for text data. The trained model, which predicts the news category, is saved as `model.pkl`.

## Frontend Development with Streamlit
A dynamic and interactive front-end interface was developed using Streamlit. This allows users to:
*   Input a news headline and (optionally) a short description and link.
*   Submit the information to the backend Naive Bayes model for classification.
*   View the predicted category and the model's accuracy.
*   Browse existing news articles conveniently organized into tabs by category.

Streamlit's ease of use enabled rapid development of a functional and visually appealing web application for showcasing the classification model.

## Project Structure
A typical structure for this project would be:


NewsPalette/
├── Home.py # Main Streamlit application script
├── NewsCategorizer.csv # Original dataset
├── model.pkl # Serialized trained Naive Bayes model
├── vector.pkl # Serialized TF-IDF vectorizer
├── preprocessed.xlsx # (Optional) Excel file of preprocessed data
├── assets/ # Folder for images
│ ├── Img-1.png
│ ├── Img-2.png
│ ├── Img-3.png
│ └── Img-4.png
├── requirements.txt # Python package dependencies
└── README.md # This file

*(Note: `new_forms.html`, `style.css`, `summa.html` from the initial file list seem to be auxiliary or experimental and are not directly part of the core Streamlit app logic in `Home.py`)*

## Setup & Installation
To set up and run this project locally:

1.  **Clone the repository (if applicable) or download the project files.**
    ```bash
    # Example: git clone https://github.com/yourusername/NewsPalette.git
    # cd NewsPalette
    ```

2.  **Create a Python virtual environment (recommended):**
    ```bash
    python -m venv venv
    # On Windows
    venv\Scripts\activate
    # On macOS/Linux
    source venv/bin/activate
    ```

3.  **Install dependencies:**
    Create a `requirements.txt` file with the following content:
    ```txt
    streamlit
    pandas
    scikit-learn
    # Add any other specific versions if necessary
    ```
    Then run:
    ```bash
    pip install -r requirements.txt
    ```

4.  **Ensure necessary files are present:**
    Make sure `Home.py`, `NewsCategorizer.csv`, `model.pkl`, and `vector.pkl` are in the root directory of the project (or adjust paths in `Home.py` accordingly).

## Usage
To run the Streamlit application:
```bash
streamlit run Home.py
IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END

Navigate to the local URL provided by Streamlit in your web browser (usually http://localhost:8501).

Image Gallery

Main interface for news input and classification:

![alt text](./assets/Img-1.png)

Display of predicted category and model accuracy:

![alt text](./assets/Img-4.png)

Example of news articles displayed under the "TRAVEL" category tab:

![alt text](./assets/Img-2.png)

Example of news articles displayed under another category tab (e.g., "FOOD & DRINK"):

![alt text](./assets/Img-3.png)

(Note: Ensure the image paths ./assets/Img-X.png are correct relative to your README.md file within your repository structure. If the images are in the root, it would be ./Img-X.png.)

Future Enhancements

Incorporate 'Description' in Prediction: Enhance the prediction model by using both the headline and description provided by the user, which might require retraining the model and TF-IDF vectorizer on combined text.

Dynamic Image Display: If image URLs are available in the dataset, display relevant images for each news card instead of static placeholders.

User Accounts & Personalization: Allow users to create accounts and save preferred news categories.

Advanced Model Evaluation: Implement more detailed model evaluation metrics (precision, recall, F1-score per category) and display a confusion matrix.

Continuous Learning: Develop a pipeline to periodically retrain the model with new data to maintain accuracy.

API Endpoint: Expose the classification model via an API for integration with other services.

IGNORE_WHEN_COPYING_START
content_copy
download
Use code with caution.
IGNORE_WHEN_COPYING_END
