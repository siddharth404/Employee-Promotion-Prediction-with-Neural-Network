import streamlit as st
import pandas as pd
import numpy as np
from sklearn.preprocessing import LabelEncoder
from keras.models import Sequential
from keras.layers import Dense, Dropout
from keras.optimizers import Adam

# Load dataset (replace these paths with actual file paths or upload feature in streamlit)
train_data = pd.read_csv('employee_train.csv')
test_data = pd.read_csv('employee_test.csv')

# Data preprocessing
le = LabelEncoder()

# Function to encode input data
def encode_data(data, columns):
    for col in columns:
        data[col] = le.fit_transform(data[col])
    return data

columns_to_encode = ['gender', 'department', 'region', 'recruitment_channel', 'education']
train_data = encode_data(train_data, columns_to_encode)
test_data = encode_data(test_data, columns_to_encode)

# Extract features and labels
col = ['department', 'region', 'education', 'gender', 'recruitment_channel', 'no_of_trainings', 
       'age', 'previous_year_rating', 'length_of_service', 'awards_won?', 'avg_training_score']
train_data_x = train_data[col]
train_data_y = train_data['is_promoted']

# Build the Neural Network model
def build_model():
    model = Sequential()
    model.add(Dense(8, input_shape=(11,), activation='relu'))
    model.add(Dropout(0.3))
    model.add(Dense(1, activation='sigmoid'))
    
    adam = Adam(learning_rate=0.001)
    model.compile(optimizer=adam, loss='binary_crossentropy', metrics=['accuracy'])
    return model

model = build_model()

# Streamlit app
st.title("Employee Promotion Prediction")

# Collect user inputs
gender = st.selectbox("Gender", ["Male", "Female"])
department = st.selectbox("Department", train_data['department'].unique())
region = st.selectbox("Region", train_data['region'].unique())
education = st.selectbox("Education", train_data['education'].unique())
recruitment_channel = st.selectbox("Recruitment Channel", train_data['recruitment_channel'].unique())
no_of_trainings = st.number_input("Number of Trainings", min_value=1, max_value=10)
age = st.number_input("Age", min_value=18, max_value=60)
previous_year_rating = st.number_input("Previous Year Rating", min_value=1.0, max_value=5.0, step=0.1)
length_of_service = st.number_input("Length of Service (in years)", min_value=1, max_value=40)
awards_won = st.selectbox("Awards Won", ["Yes", "No"])
avg_training_score = st.number_input("Average Training Score", min_value=0, max_value=100)

# Encode the inputs
input_data = pd.DataFrame({
    'department': [le.fit_transform([department])[0]],
    'region': [le.fit_transform([region])[0]],
    'education': [le.fit_transform([education])[0]],
    'gender': [le.fit_transform([gender])[0]],
    'recruitment_channel': [le.fit_transform([recruitment_channel])[0]],
    'no_of_trainings': [no_of_trainings],
    'age': [age],
    'previous_year_rating': [previous_year_rating],
    'length_of_service': [length_of_service],
    'awards_won?': [1 if awards_won == 'Yes' else 0],
    'avg_training_score': [avg_training_score]
})

# Predict promotion
if st.button('Predict Promotion'):
    prediction = model.predict(input_data)
    prediction = (prediction > 0.5).astype(int)
    if prediction == 1:
        st.success("The employee is likely to be promoted.")
    else:
        st.warning("The employee is not likely to be promoted.")
