# Video-Game-Capstone

A repository for the analysis of the video game industry. This is based off of data collected from [RAWG](https://rawg.io/). An application that predicts how well a video game would rate among people who play video games based on the criteria chosen.



## Built With

**Exploration/Cleaning:** Jupyter, Python

**Front-End:** Streamlit

Python is used to tackle pulling from the API, cleaning, and setting up the streamlit app. Notebooks were used to assess data for relevancy and transform some data. An XGBoost classifier is applied to the inputs to predict the rating.


## Features

- Predictive video game rating using XGBoost.
- Output table of feature importance.
- Searchable attributes of various kinds.


## Roadmap

- Additional metrics for better understanding of the data.

- Additional selection criteria for better training.


## Used By

This project is for video game developers that are looking to build a game, but are uncertain what type of game will sell well within the current/recent market. The app is designed to predict a rating so that developers know what to add to their game that gamers will enjoy.



## Lessons Learned

Cleaning the data could have been smoother. Nesting jsons were cumbersome to tease out the correct data, but once some functions were complete the process became easier. Learning streamlit for the first time had a decent learning curve, but also became easier over time. Displaying feature importance is still a challenge, but one I plan to overcome.


## Acknowledgements
 - [RAWG Website](https://rawg.io/)
 - [Awesome Readme Templates](https://awesomeopensource.com/project/elangosundar/awesome-README-templates)
 - [Awesome README](https://github.com/matiassingers/awesome-readme)
 - [How to write a Good readme](https://bulldogjob.com/news/449-how-to-write-a-good-readme-for-your-github-project)
