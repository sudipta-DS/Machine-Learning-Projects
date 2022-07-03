import numpy as np
import pandas as pd
import pickle
import streamlit as st

teams = ['Sunrisers Hyderabad', 'Royal Challengers Bangalore',
       'Kolkata Knight Riders', 'Kings XI Punjab', 'Delhi Capitals',
       'Mumbai Indians', 'Chennai Super Kings', 'Rajasthan Royals']

cities = ['Hyderabad', 'Bangalore', 'Mumbai', 'Indore', 'Kolkata', 'Delhi',
       'Chandigarh', 'Jaipur', 'Chennai', 'Cape Town', 'Port Elizabeth',
       'Durban', 'Centurion', 'East London', 'Johannesburg', 'Kimberley',
       'Bloemfontein', 'Ahmedabad', 'Cuttack', 'Nagpur', 'Dharamsala',
       'Visakhapatnam', 'Pune', 'Raipur', 'Ranchi', 'Abu Dhabi',
       'Sharjah', 'Mohali', 'Bengaluru']

pipe = pickle.load(open('./pipe.pkl','rb'))
st.title('IPL WIN PREDICTOR')

COL1 ,COL2 = st.columns(2)

with COL1 :
       battingteam = st.selectbox('Select batting team',sorted(teams))
with COL2 :
       bowlingteam = st.selectbox('Select bowling team',sorted(teams))

city = st.selectbox('Select City',sorted(cities))

target = st.number_input('Target')

COL3 , COL4 ,COL5 = st.columns(3)

with COL3 :
       curr_score = st.number_input('Enter Live Score')
with COL4 :
       over_played = st.number_input('Overs Played')
with COL5 :
       wickets_fallen = st.number_input('Wickets Fallen')

if st.button('PREDICT PROBABILITY') :
       runs_left = target - curr_score
       balls_left = 120 - (6*over_played)
       wickets_left = 10 -wickets_fallen
       curr_run_rate = curr_score / over_played
       req_run_rate = (runs_left * 6) / balls_left

       input_df = pd.DataFrame({'batting_team':[battingteam], 'bowling_team': [bowlingteam],
                                'city':[city], 'runs_left':[runs_left], 'balls_left':[balls_left],
                            'wickets_left':[wickets_left], 'total_runs_x':[target], 'cur_run_rate':[curr_run_rate],
                            'req_run_rate':[req_run_rate]})
       result = pipe.predict_proba(input_df)
       lossprob = result[0][0]
       winprob = result[0][1]

       st.header(battingteam +"- "+str(round(winprob*100))+"%")
       st.header(bowlingteam +"- "+str(round(lossprob*100))+"%")
