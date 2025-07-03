import streamlit as st
import pandas as pd
import numpy as np
import joblib
import sklearn
import shap
import xgboost as xgb
import pickle
import matplotlib.pyplot as plt
import plotly.graph_objects as go
import os

explainer = joblib.load('../models/explainer_v1.joblib')
explanation = joblib.load('../models/explanation_v1.joblib')
model = joblib.load('../models/xgboostclassifier_v1.joblib')
starting_predict_df = pd.read_csv('../data/zero_df.csv')

st.set_page_config(layout="wide")
st.title('Developer Buddy')

tab1, tab2 = st.tabs(["Background", "App"])

# Side Bar Tab ----------------------------------------------------------------------------------------------------------------------------------
with tab2:

    # inputing data -----------------------------------------------------------------------------------------------------------------------------
    cleaned_data = pd.read_csv('../data/rawg_cleaned_games_no_dev_data.csv')
    genre_df = pd.read_csv('../data/genre_list.csv')
    tags_df = pd.read_csv('../data/tag_list.csv')
    stores_df = pd.read_csv('../data/stores_list.csv')
    platform_df = pd.read_csv('../data/platforms_list.csv')
    esrb_rating = ['Mature', 'Everyone 10+', 'Teen', 'Adults Only', 'Everyone', 'Rating Pending']

    # setting up streamlit objects --------------------------------------------------------------------------------------------------------------

    # Genre Checklist
    st.sidebar.title('Select Your Options')

    st.sidebar.header('Select A Genre')
    if 'genre_list' not in st.session_state.keys():
        genre_list = list(genre_df['name'])
        st.session_state['genre_list'] = genre_list
    else:
        genre_list = st.session_state['genre_list']

    selected_genres = st.sidebar.multiselect(
        label='Search or select genres',
        options=genre_list,
        default=[]
    )

    # Platform Checklist
    st.sidebar.header('Select A Platform')
    if 'platform_list' not in st.session_state.keys():
        platform_list = list(platform_df['name'])
        st.session_state['platform_list'] = platform_list
    else:
        platform_list = st.session_state['platform_list']

    selected_platforms = st.sidebar.multiselect(
        label='Search or select desired platforms',
        options=platform_list,
        default=[]
    )

    # Stores Checklist
    st.sidebar.header('Select Your Stores')
    if 'stores_list' not in st.session_state:
        stores_list = list(stores_df['name'])
        st.session_state['stores_list'] = stores_list
    else:
        stores_list = st.session_state['stores_list']

    selected_stores = st.sidebar.multiselect(
        label='Search or select desired store',
        options=stores_list,
        default=[]
    )

    # tags Checklist
    st.sidebar.header('Select Your Attributes')

    if 'tags_list' not in st.session_state:
        tags_list = list(tags_df['name'])
        st.session_state['tags_list'] = tags_list
    else:
        tags_list = st.session_state['tags_list']

    selected_tags = st.sidebar.multiselect(
        label='Search or select desired attributes',
        options=tags_list,
        default=[]
    )

    # ESRB Checklist
    st.sidebar.header('Select an ESRB Rating')

    if 'esrb_rating' not in st.session_state.keys():
        st.session_state['esrb_rating'] = esrb_rating
    else:
        esrb_rating = st.session_state['esrb_rating']

    selected_esrb = st.sidebar.selectbox(
        label='Search or select an ESRB Rating',
        options=esrb_rating,
        index=0
    )

    # modeling ----------------------------------------------------------------------------------------------------------------------------------

    col = st.columns(2, gap='medium')

    X = starting_predict_df.drop(columns=['Unnamed: 0'])

    # with col[0]:
    #     st.header('Explanation: What You Selected')
    #     st.write(f'{selected_genres}')


    with col[0]:
        st.header('Model Output')
        if len(selected_genres) > 0:
            for i in selected_genres:
                X[i + '_genre'] = 1

        if len(selected_platforms) > 0:
            for i in selected_platforms:
                X[i + '_platform'] = 1

        if len(selected_tags) > 0:
            for i in selected_tags:
                X[i] = 1

        if len(selected_stores) > 0:
            for i in selected_stores:
                X[i + '_store'] = 1

        if len(selected_esrb) > 0:
            X['esrb_rating'] = selected_esrb

        if (X == 0).all(axis=1).all():
            st.write('Select More Attributes!')
            #st.metric(label="Predicted Rating", value=0)
        else:   
            y_pred = model.predict(X)
            if y_pred == 1:
                st.write('Model predicts mostly negative :persevere:')
                st.plotly_chart(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = int(y_pred[0]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Rating"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "red"},
                        {'range': [1.5, 2.5], 'color': "orange"},
                        {'range': [2.5, 3.5], 'color': "yellow"},
                        {'range': [3.5, 4.5], 'color': "green"}
                    ],
                    'bar': {
                    'color': "black",
                    }
                    },
                    
                    ))
                )
            elif y_pred == 2:
                st.write('Model predicts negative :neutral_face:')
                st.plotly_chart(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = int(y_pred[0]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Rating"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "red"},
                        {'range': [1.5, 2.5], 'color': "orange"},
                        {'range': [2.5, 3.5], 'color': "yellow"},
                        {'range': [3.5, 4.5], 'color': "green"}
                    ],
                    'bar': {
                    'color': "black",
                    }
                    }))
                )
            elif y_pred == 3: 
                st.write('Model predicts positive :slightly_smiling_face:')
                st.plotly_chart(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = int(y_pred[0]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Rating"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "red"},
                        {'range': [1.5, 2.5], 'color': "orange"},
                        {'range': [2.5, 3.5], 'color': "yellow"},
                        {'range': [3.5, 4.5], 'color': "green"}
                    ],
                    'bar': {
                    'color': "black",
                    }
                    }))
                )
            else:
                st.write('Model predicts very positive :smiley:')
                st.plotly_chart(go.Figure(go.Indicator(
                mode = "gauge+number",
                value = int(y_pred[0]),
                domain = {'x': [0, 1], 'y': [0, 1]},
                title = {'text': "Rating"},
                gauge={
                    'axis': {'range': [None, 5]},
                    'steps': [
                        {'range': [0.5, 1.5], 'color': "red"},
                        {'range': [1.5, 2.5], 'color': "orange"},
                        {'range': [2.5, 3.5], 'color': "yellow"},
                        {'range': [3.5, 4.5], 'color': "green"}
                    ],
                    'bar': {
                    'color': "black",
                    }
                    }))
                )

        

    with col[1]:
        st.header('Feature Importance')

        y_pred = model.predict(X)
        target_list = list(model['xgboost'].classes_)
        probs = model.predict_proba(X)
        target = y_pred
        class_index = target_list.index(target)
        i = 0

        explanation = explainer(model[:-2].transform(X))

        fig, ax = plt.subplots()
        shap.plots.waterfall(explanation[i, :, class_index], show=False)
        st.pyplot(fig)

        feature_names = model[:-2].get_feature_names_out()
        importances = model['xgboost'].feature_importances_
        importance_df = pd.DataFrame({
            'Feature': feature_names,
            'Importance': importances
        }).sort_values(by='Importance', ascending=False)
        st.write(importance_df.reset_index(drop=True))

# Explanation Tab --------------------------------------------------------------------------------------------------------------------------------------------------
with tab1:
    st.subheader('Overview')
    st.write('When developing a video game, a developer must use certain aspects in order to build that game. In todays market, some options may result in a higher rating with gamers than others. This app predicts whether those options chosen will do well within todays market.')

    st.header('Data Processing Explanation')
    st.write('Some explanation is warranted for how this data is gathered, cleaned and modeled.' \
    'The data is acquired from RAWG.io via API calls. I called 2500 pages of the games portion of the code to retrieve all of the data I needed. The API call is pretty straightforward without needing time breaks and the like.' \
    'The data itself had multiple layers of json code for each of the desired columns. I set up a general functiont that needed tweaking between each of the data columns. For example, using he fucntion would look for one name of the column when the actual name might be slightly different than a different data column.' \
    'The function I used converted the json to columns of data with either a zero or one, showing categorically if the column name exists for that particular game. The only exception to that rule was the ESRB rating column, which I left as strings for the OneHotEncoder to work out.' \
    'This took the most amount of time as gathering all of that data into a useful dataframe to be modeled was a challenge.' \
    'I then pass the data through a pipeline containing OneHotEncoder and a Simple Imputer using the most_frequent strategy. I then used XGBoostClassifier as my model to predict outcomes' \
    'This resulted in data heavily skewed towards the bad ratings, so I then applied SMOTE before the XGBoost model to beef up the higher rankings.' \
    'I then created the app to apply the model within.')
    