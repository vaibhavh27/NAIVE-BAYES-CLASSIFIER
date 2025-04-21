import streamlit as st
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import plotly.express as px
from plotly.subplots import make_subplots
import plotly.graph_objs as go
import nltk
from wordcloud import WordCloud
import numpy as np
from PIL import Image
import cv2


st.set_page_config(layout="wide")

data = pd.read_excel("preprocessed.xlsx")

fig = px.pie(data, names='category', title='Category Distribution')

st.title('Category Distribution Pie Chart')
st.plotly_chart(fig)

image_names = [
        "Business",
        "Entertainment",
        "Food And Drinks",
        "Politics",
        "Parenting",
        "Sports",
        "Style",
        "Travel",
        "Wellness",
        "World"
    ]

    # Create a selectbox for image selection
selected_image_name = st.selectbox(" ", image_names)


# image_paths = [
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_bussiness.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_entertainment.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_food.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_politics.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_praenting.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_sports.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_style.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_travel.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_wellness.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\\top_keyword_world.png")
#     ]
    



image_paths = [
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_bussiness.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_entertainment.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_food.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_politics.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_praenting.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_sports.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_style.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_travel.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_wellness.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_world.png")
    ]




selected_image_path = image_paths[image_names.index(selected_image_name)]

    # Display the selected image
st.image(selected_image_path, use_column_width=True)


selected_image_name = st.selectbox("  ", image_names)


# image_paths = [
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_business.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_entertainment.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_food.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_politics.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_parenting.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_sports.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_style.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_travel.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_wellness.png"),
#         Image.open("D:\College\SEM-5\ML_HACKATON\code\pages\word_colud_world.png")
#     ]
    


image_paths = [
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_bussiness.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_entertainment.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_food.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_politics.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_praenting.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_sports.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_style.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_travel.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_wellness.png"),
    Image.open("C:/Users/Vaibhav/Downloads/NewsPalette-main/pages/top_keyword_world.png")
]

selected_image_path = image_paths[image_names.index(selected_image_name)]

    # Display the selected image
st.image(selected_image_path, use_column_width=True)

