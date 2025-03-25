import os
from pickle import load
import streamlit as st
import sklearn
import pandas as pd
import numpy as np
from xgboost import XGBClassifier



model = load(open("/workspaces/Kickstarter-AC-KR-PN-App/src-final/XGBoost.sav", "rb"))
ordinal_encoder = load(open("/workspaces/Kickstarter-AC-KR-PN-App/src-final/Ord_Encoder.sav", "rb")) 

class_dict = {
    "0": "Failure",
    "1": "Success",
}

month_dict = {
    'January': '2024-01-01',
    'Febuary': '2024-02-01',
    'March': '2024-03-01',
    'April': '2024-04-01',
    'May': '2024-05-01',
    'June': '2024-06-01',
    'July': '2024-07-01',
    'August': '2024-08-01',
    'September': '2024-09-01',
    'October': '2024-10-01',
    'November': '2024-11-01',
    'December': '2024-12-01'
}

# feature_columns = ['country_displayable_name', 'goal', 'month', 'category_name', 'subcategory', 'city']  

country = st.selectbox(
    "Which country do you live in?",
    ("the United States", "the United Kingdom", "Canada", "Australia", "Germany", "Mexico", "France", "Italy", "Spain", "Hong Kong", "the Netherlands", "Sweden", "Japan", "Singapore", "Denmark", "New Zealand", "Switzerland", "Belgium", "Ireland", "Austria", "Poland", "Norway", "Greece", "Luxembourg", "Slovenia"),
    index=None,
    placeholder="Select country"
)

goal = int(st.number_input("Enter your goal in USD"))

month_input = "January"

month_input = st.selectbox(
    "Which month do you plan to launch your campaign in?",
    ("January", "Febuary", "March", "April", "May", "June", "July", "August", "September", "October", "November", "December"),
    index=None,
    placeholder="January"
)



category = st.selectbox(
    "Select the category of your campaign",
    ('Games', 'Comics', 'Music', 'Design', 'Art', 'Publishing', 'Food',
       'Fashion', 'Technology', 'Film & Video', 'Dance',
       'Photography', 'Theater', 'Journalism', 'Crafts'),
    index=None,
    placeholder="Games"
)

if category == "Games":
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Playing Cards', 'Tabletop Games', 'Video Games',
       'Gaming Hardware', 'Mobile Games', 'Puzzles', 'Live Games'),
    index=None,
    placeholder='Tabletop Games'
)

elif category == "Comics":
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Comic Books', 'Graphic Novels', 'Webcomics', 'Anthologies',
       'Events'),
    index=None,
    placeholder='Comic Books'
)
    
elif category == 'Music':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('World Music', 'Electronic Music', 'Faith', 'Metal',
       'Country & Folk', 'Pop', 'Rock', 'Classical Music', 'Indie Rock',
       'Hip-Hop', 'Jazz', 'Blues', 'Punk', 'Kids', 'R&B', 'Latin',
       'Chiptune', 'Comedy'),
    index=None,
    placeholder='World Music'
)
    
elif category == 'Design':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Toys', 'Graphic Design', 'Product Design', 'Civic Design',
       'Interactive Design', 'Architecture', 'Typography'),
    index=None,
    placeholder='Toys'
)
    
elif category == 'Art':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Performance Art', 'Installations', 'Sculpture', 'Illustration',
       'Ceramics', 'Painting', 'Public Art', 'Textiles', 'Digital Art',
       'Conceptual Art', 'Mixed Media', 'Social Practice', 'Video Art'),
    index=None,
    placeholder='Performance Art'
)
    
elif category == 'Publishing':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Nonfiction', 'Radio & Podcasts', 'Academic', 'Literary Journals',
       'Periodicals', 'Young Adult', 'Poetry', 'Fiction', 'Anthologies',
       'Translations', 'Zines', 'Art Books', "Children's Books",
       'Calendars', 'Comedy', 'Literary Spaces', 'Letterpress'),
    index=None,
    placeholder='Nonfiction'
)
    
elif category == 'Food':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Spaces', 'Drinks', 'Small Batch', 'Restaurants', 'Vegan',
       "Farmer's Markets", 'Community Gardens', 'Cookbooks', 'Events',
       'Bacon', 'Farms', 'Food Trucks'),
    index=None,
    placeholder='Spaces'
)
    
elif category == 'Fashion':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Ready-to-wear', 'Accessories', 'Childrenswear', 'Footwear',
       'Couture', 'Jewelry', 'Apparel', 'Pet Fashion'),
    index=None,
    placeholder='Apparel'
)

elif category == 'Technology':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Software', 'Apps', 'Hardware', '3D Printing', 'Robots', 'Gadgets',
       'Sound', 'Web', 'Flight', 'Camera Equipment', 'Makerspaces',
       'Wearables', 'DIY Electronics', 'Space Exploration',
       'Fabrication Tools'),
    index=None,
    placeholder='Software'
)
    
elif category == 'Film & Video':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Television', 'Narrative Film', 'Science Fiction', 'Family',
       'Thrillers', 'Webseries', 'Documentary', 'Shorts', 'Comedy',
       'Music Videos', 'Animation', 'Horror', 'Festivals', 'Drama',
       'Action', 'Movie Theaters', 'Fantasy', 'Romance', 'Experimental'),
    index=None,
    placeholder='Television'
)
    
elif category == 'Dance':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Performances', 'Workshops', 'Spaces', 'Residencies'),
    index=None,
    placeholder='Performances'
)
    
elif category == 'Photography':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Photobooks', 'Fine Art', 'People', 'Nature', 'Animals', 'Places'),
    index=None,
    placeholder='Photobooks'
)
    
elif category == 'Theater':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Plays', 'Musical', 'Experimental', 'Comedy', 'Festivals',
       'Immersive', 'Spaces'),
    index=None,
    placeholder='Plays'
)
    
elif category == 'Journalism':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Web', 'Audio', 'Print', 'Photo', 'Video'),
    index=None,
    placeholder='Web'
)
    
elif category == 'Crafts':
    subcategory = st.selectbox(
    "Select the subcategory of your campaign",
    ('Printing', 'Stationery', 'Woodworking', 'DIY', 'Pottery',
       'Knitting', 'Candles', 'Crochet', 'Embroidery', 'Glass', 'Quilts',
       'Taxidermy', 'Weaving'),
    index=None,
    placeholder='Printing'
)
    
city = st.text_input("Enter the name of your city, followed by state/province, separated by a comma", "Enter City Here")

if st.button("Predict"):
    month = str(month_dict[month_input])
    features = [country, month, category, subcategory, city]  
    features = np.array(features)
    features = features.reshape(1,-1)
    try:
        encoded_features = ordinal_encoder.transform(features)
    except ValueError:
        st.write("Try reformatting the name of the city or filling any empty values.")
    final_features = np.hstack((encoded_features, [[goal]]))
    prediction = str(model.predict(final_features)[0])
    pred_class = class_dict[prediction]
    st.write("Prediction:", pred_class)
