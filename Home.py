import streamlit as st
import pandas as pd
import pickle
from bs4 import BeautifulSoup
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score
from flask import Flask, request
from streamlit_option_menu import option_menu

st.set_page_config(layout="wide")

# Sample news data for demonstration
news_data = pd.read_csv("NewsCategorizer.csv")
data = pd.read_csv("NewsCategorizer.csv")

# Train a Naive Bayes classifier on the sample data (you should train your model on a real dataset)
tfidf_vectorizer = TfidfVectorizer(max_features=5000)
X = tfidf_vectorizer.fit_transform(news_data['headline'])
y = news_data['category']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

X1 = data.drop('category',axis=1)
X1 = X1.drop('keywords',axis=1)
y1 = data['category']

X_train_1, X_test_1, y_train_1, y_test_1 = train_test_split(X1, y1, test_size=0.2, random_state=42)

naive_bayes = MultinomialNB()
naive_bayes.fit(X_train, y_train)

pickle_file_name = "model.pkl"
with open(pickle_file_name, 'rb') as pickle_file:
    models = pickle.load(pickle_file)
    
    
vector_file_name = "vector.pkl"
with open(vector_file_name, 'rb') as vector_file:
    vector = pickle.load(vector_file)

print(models)
# Calculate accuracy
y_pred = naive_bayes.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)


styles = """
    /* Center align text */
    .center {
        text-align: center;
    }

    /* Add a shadow to the box */
    .shadow {
        box-shadow: 0px 0px 10px rgba(0, 0, 0, 0.25);
    }

    /* Add a gradient background */
    body {
        background: linear-gradient(to bottom, #ffffff, #f2f2f2);
    }

    /* Style the header */
    .header {
        background: linear-gradient(to bottom, #000000, #2C3539);
        color: white;
        padding: 20px;
    }

    /* Style the footer */
    .footer {
        background: linear-gradient(to bottom, #4B0082, #800080);
        color: white;
        padding: 20px;
    }
"""
st.markdown(f'<style>{styles}</style>', unsafe_allow_html=True)
header_html = """
    <div class="header shadow">
        <h1 class="center rainbow-text">News Classification App</h1>
    </div>
"""
st.markdown(header_html, unsafe_allow_html=True)



# Custom HTML and CSS for news container
container_html = """
<style>
:root {
  --red: #ef233c;
  --darkred: #c00424;
  --platinum: #e5e5e5;
  --black: #2b2d42;
  --white: #fff;
  --thumb: #edf2f4;
}
* {
  box-sizing: border-box;
  padding: 0;
  margin: 0;
}
body {
  font: 16px / 24px "Rubik", sans-serif;
  color: var(--white);
  background: var(--black);
  margin: 50px 0;
}
.container {
  width: 1200px;
  height:600px;
  padding: 0 15px;
  margin: 0 auto;
}

.new-container{
  width: 1200px;
  height:700px;
  padding: 0 15px;
  margin: 0 auto;
}

h2 {
  font-size: 50px;
  margin-bottom: 1em;
  text-align : center;
}
.cards {
  width: 1400px;
  height:500px;
  display: flex;
  padding: 25px 0px;
  list-style: none;
  scroll-snap-type: x mandatory;
}

.new-cards{
  width: 1075px;
  height:600px;
  display: flex;
  padding: 25px 0px;
  list-style: none;
  scroll-snap-type: x mandatory;
  text-align : center;
}


.card {
  width: 1200px;
  height:480px;
  display: flex;
  overflow: hidden;
  flex-direction: column;
  flex: 0 0 100%;
  padding: 20px;
  background: #0E1117;
  border-radius: 12px;
  box-shadow: 0 5px 10px white;
  scroll-snap-align: start;
  transition: all 0.2s;
  text-align: center;
}

#new-card{
  width: 1000px;
  height:600px;
  display: flex;
  overflow: hidden;
  flex-direction: column;
  flex: 0 0 100%;
  padding: 20px;
  background: #0E1117;
  border-radius: 12px;
  box-shadow: 0 5px 10px white;
  scroll-snap-align: start;
  transition: all 0.2s;
  
}

.card img {
			width: 300px;
			height: 150px;
			object-fit: cover;
			margin-bottom: 7px;
			border-radius: 4px;
}

.card:not(:last-child) {
  margin-right: 10px;
}
.card:hover {
  color: var(--white);
  background:#fe8267;
}
.card .card-title {
  font-size: 20px;
}
.card .card-content {
  margin: 20px 0;
  max-width: 85%;
}
.card .card-link-wrapper {
  margin-top: auto;
}
.card .card-link {
  display: inline-block;
  text-decoration: none;
  color: white;
  background: var(--red);
  padding: 6px 12px;
  border-radius: 8px;
  transition: background 0.2s;
}
.card:hover .card-link {
  background: var(--darkred);
}
.cards::-webkit-scrollbar {
  height: 12px;
}
.cards::-webkit-scrollbar-thumb,
.cards::-webkit-scrollbar-track {
  border-radius: 92px;
}
.cards::-webkit-scrollbar-thumb {
  background: var(--darkred);
}
.cards::-webkit-scrollbar-track {
  background: var(--thumb);
}
@media (width: 200%) {
  .card {
    flex-basis: calc(50% - 10px);
  }
  .card:not(:last-child) {
    margin-right: 20px;
  }
}
@media (min-width: 700px) {
  .card {
    flex-basis: calc(calc(100% / 3) - 20px);
  }
  .card:not(:last-child) {
    margin-right: 30px;
  }
}
@media (min-width: 1100px) {
  .card {
    flex-basis: calc(25% - 30px);
  }
  .card:not(:last-child) {
    margin-right: 40px;
  }
}
</style>
"""
st.markdown(container_html , unsafe_allow_html=True)


# Create tabs for different news categories
tab, tab1, tab2, tab3, tab4, tab5, tab6, tab7, tab8, tab9, tab10 = st.tabs(["PREDICTION" ,"TRAVEL", "FOOD & DRINK", "ENTERTAINMENT", "WORLD NEWS", "WELLNESS","POLITICS","STYLE & BEAUTY","PARENTING","BUSINESS","SPORTS"])



with tab:
  # Create placeholders for user input
  Headline = st.text_input('Headline:')
  Description = st.text_area('Description:')
  Link = st.text_input('Link:')

  prediction = None  # Initialize prediction outside the button click scope

  if st.button("Classify"):
      if Headline:
        # Preprocess the user input
          news_text = pd.Series(Headline)

        # Vectorize the input
          X_test = tfidf_vectorizer.transform(news_text)

        # Make predictions
          prediction = naive_bayes.predict(X_test)[0]

          st.markdown(
            """
            <style>
            .centered-title {
                text-align: center;
                font-size: 28px;
            }
            
            h2{
              text-align : center;
              font-size: 24px;
            }
            
            </style>
            """,
            unsafe_allow_html=True
          )

          st.markdown("<h1 class='centered-title'>News Category:</h1>", unsafe_allow_html=True)
          html_code = f"<h2>{prediction}</h2>"
          st.markdown(html_code, unsafe_allow_html=True)


# Display the accuracy
  st.markdown(
    """
    <style>
    .centered-title {
        text-align: center;
        font-size: 28px;
    }
    
    h2{
      text-align : center;
      font-size: 24px;
    }
    
    </style>
    """,
    unsafe_allow_html=True
  )

  st.markdown("<h1 class='centered-title'>Model Accuracy:</h1>", unsafe_allow_html=True)
  html_code = f"<h2>{accuracy:.2%}</h2>"
  st.markdown(html_code, unsafe_allow_html=True)
  

# Display all sports news from the test set under the "Sports" tab
with tab1:
    df = X_train_1[y_train_1 == "TRAVEL"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "TRAVEL":
      st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIWzrdHLuRWUjbOdbUdjq--bhKvG-oEi43mA&usqp=CAU" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIWzrdHLuRWUjbOdbUdjq--bhKvG-oEi43mA&usqp=CAU" style="display: block; margin: 0 auto;"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIWzrdHLuRWUjbOdbUdjq--bhKvG-oEi43mA&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRIWzrdHLuRWUjbOdbUdjq--bhKvG-oEi43mA&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break 

with tab2:
    df = X_train_1[y_train_1 == "FOOD & DRINK"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "FOOD & DRINK":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://d12oja0ew7x0i8.cloudfront.net/images/Article_Images/ImageForArticle_18224_4482733283598383581.jpg" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://d12oja0ew7x0i8.cloudfront.net/images/Article_Images/ImageForArticle_18224_4482733283598383581.jpg"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://d12oja0ew7x0i8.cloudfront.net/images/Article_Images/ImageForArticle_18224_4482733283598383581.jpg"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://d12oja0ew7x0i8.cloudfront.net/images/Article_Images/ImageForArticle_18224_4482733283598383581.jpg"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
with tab3:
    
    df = X_train_1[y_train_1 == "ENTERTAINMENT"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "ENTERTAINMENT":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq0X84cInUnKnXjn1l6hzk_dkpOz7ykrkNLKQI0NWVbMw_0Sl2pihrqX8NJVUEclzlLCo&usqp=CAU" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq0X84cInUnKnXjn1l6hzk_dkpOz7ykrkNLKQI0NWVbMw_0Sl2pihrqX8NJVUEclzlLCo&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq0X84cInUnKnXjn1l6hzk_dkpOz7ykrkNLKQI0NWVbMw_0Sl2pihrqX8NJVUEclzlLCo&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcSq0X84cInUnKnXjn1l6hzk_dkpOz7ykrkNLKQI0NWVbMw_0Sl2pihrqX8NJVUEclzlLCo&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break

    
with tab4:
    
    df = X_train_1[y_train_1 == "WORLD NEWS"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "WORLD NEWS":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ILw1FfxNMIHvDuy3v2kteMFVyTYq8eLKXQ&usqp=CAU" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ILw1FfxNMIHvDuy3v2kteMFVyTYq8eLKXQ&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ILw1FfxNMIHvDuy3v2kteMFVyTYq8eLKXQ&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcS5ILw1FfxNMIHvDuy3v2kteMFVyTYq8eLKXQ&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
with tab5:
    
    df = X_train_1[y_train_1 == "WELLNESS"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "WELLNESS":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://media.istockphoto.com/id/1208604845/vector/healthy-lifestyle-and-self-care-concept.jpg?s=612x612&w=0&k=20&c=4RXl4xGUFpQWHf_LVBRngZRsikqw8BOc51poaItPxMU=" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://media.istockphoto.com/id/1208604845/vector/healthy-lifestyle-and-self-care-concept.jpg?s=612x612&w=0&k=20&c=4RXl4xGUFpQWHf_LVBRngZRsikqw8BOc51poaItPxMU="><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://media.istockphoto.com/id/1208604845/vector/healthy-lifestyle-and-self-care-concept.jpg?s=612x612&w=0&k=20&c=4RXl4xGUFpQWHf_LVBRngZRsikqw8BOc51poaItPxMU="><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://media.istockphoto.com/id/1208604845/vector/healthy-lifestyle-and-self-care-concept.jpg?s=612x612&w=0&k=20&c=4RXl4xGUFpQWHf_LVBRngZRsikqw8BOc51poaItPxMU="><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
        
with tab6:
    
    df = X_train_1[y_train_1 == "POLITICS"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "POLITICS":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP9C0COLIbuDGNsFes5iJIziQegWnUG_A5gKjWzsD-3ow4zesN-E-ihyWtDNHeY4Fvsw0&usqp=CAU" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP9C0COLIbuDGNsFes5iJIziQegWnUG_A5gKjWzsD-3ow4zesN-E-ihyWtDNHeY4Fvsw0&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP9C0COLIbuDGNsFes5iJIziQegWnUG_A5gKjWzsD-3ow4zesN-E-ihyWtDNHeY4Fvsw0&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://encrypted-tbn0.gstatic.com/images?q=tbn:ANd9GcRP9C0COLIbuDGNsFes5iJIziQegWnUG_A5gKjWzsD-3ow4zesN-E-ihyWtDNHeY4Fvsw0&usqp=CAU"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
        
with tab7:
    
    df = X_train_1[y_train_1 == "STYLE & BEAUTY"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "STYLE & BEAUTY":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://img.huffingtonpost.com/asset/6511bb9f2400005300538759.jpeg?cache=0irx4LM6lf&ops=800_450" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://img.huffingtonpost.com/asset/6511bb9f2400005300538759.jpeg?cache=0irx4LM6lf&ops=800_450"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://img.huffingtonpost.com/asset/6511bb9f2400005300538759.jpeg?cache=0irx4LM6lf&ops=800_450"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://img.huffingtonpost.com/asset/6511bb9f2400005300538759.jpeg?cache=0irx4LM6lf&ops=800_450"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
        
with tab8:
    
    df = X_train_1[y_train_1 == "PARENTING"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "PARENTING":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://images.ctfassets.net/hrltx12pl8hq/2r7WVYg84lwLgcUl7GWVe2/ce027d3996670c82202a578cf4d57412/parenthood-images.jpg?fit=fill&w=600&h=400" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://images.ctfassets.net/hrltx12pl8hq/2r7WVYg84lwLgcUl7GWVe2/ce027d3996670c82202a578cf4d57412/parenthood-images.jpg?fit=fill&w=600&h=400"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://images.ctfassets.net/hrltx12pl8hq/2r7WVYg84lwLgcUl7GWVe2/ce027d3996670c82202a578cf4d57412/parenthood-images.jpg?fit=fill&w=600&h=400"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://images.ctfassets.net/hrltx12pl8hq/2r7WVYg84lwLgcUl7GWVe2/ce027d3996670c82202a578cf4d57412/parenthood-images.jpg?fit=fill&w=600&h=400"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
    
with tab9:
    
    df = X_train_1[y_train_1 == "BUSINESS"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "BUSINESS":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://t3.ftcdn.net/jpg/05/63/66/48/360_F_563664874_ibw1AUSzTgJ4vUz3WxTehvTSC53FVLJB.jpg" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://t3.ftcdn.net/jpg/05/63/66/48/360_F_563664874_ibw1AUSzTgJ4vUz3WxTehvTSC53FVLJB.jpg"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://t3.ftcdn.net/jpg/05/63/66/48/360_F_563664874_ibw1AUSzTgJ4vUz3WxTehvTSC53FVLJB.jpg"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://t3.ftcdn.net/jpg/05/63/66/48/360_F_563664874_ibw1AUSzTgJ4vUz3WxTehvTSC53FVLJB.jpg"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
        
with tab10:
    
    df = X_train_1[y_train_1 == "SPORTS"]
    df = df.reset_index(drop=True)
    headline = 'headline'
    description = 'short_description'
    link = 'links'
    dot='...'
    j = 0
    
    if prediction == "SPORTS":
        st.markdown(f'<div class="new-container"><ul class="new-cards"><li class="card" id = "new-card"><img src="https://thumbs.dreamstime.com/b/sport-equipment-2-22802518.jpg" style="display: block; margin: 0 auto;width: 500px; height:300px"><div><br><h4 class="card-title">{Headline}</h4><div class="card-content"><p>{Description}</p></div></div><div class="card-link-wrapper"><a href={Link} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)

    
    for i in range(0,len(df),3):
        j += 1
        des_1 = df[description][i]
        des_2 = df[description][i+1]
        des_3 = df[description][i+2]
        
        st.markdown(f'<div class="container"><ul class="cards"><li class="card"><img src="https://thumbs.dreamstime.com/b/sport-equipment-2-22802518.jpg"><div><br><h4 class="card-title">{df[headline][i]}</h4><div class="card-content"><p>{des_1[:100]+dot}</p></div></div><div class="card-link-wrapper"><a href={df[link][i]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://thumbs.dreamstime.com/b/sport-equipment-2-22802518.jpg"><div><br><h4 class="card-title">{df[headline][i+1]}</h4><div class="card-content"><p>{des_2[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+1]} target = "_self" class="card-link">Learn More</a></div></li><li class="card"><img src="https://thumbs.dreamstime.com/b/sport-equipment-2-22802518.jpg"><div><br><h4 class="card-title">{df[headline][i+2]}</h4><div class="card-content"><p>{des_3[:100]}</p></div></div><div class="card-link-wrapper"><a href={df[link][i+2]} target = "_self" class="card-link">Learn More</a></div></li></ul></div>', unsafe_allow_html=True)
        if(j == 5):
            break
    
    
    
    