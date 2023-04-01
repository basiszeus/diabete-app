# IMPORT STATEMENTS

import streamlit as st
import pandas as pd
from PIL import Image
import numpy as np
from sklearn.metrics import accuracy_score
import matplotlib.pyplot as plt
import seaborn as sns
import pickle
from sklearn.model_selection import train_test_split

# PAGE CONFIGURATION
st.set_page_config(
    page_title="Diabetes Screening",
    page_icon="https://ssm.swiss/wp-content/uploads/2019/09/SSM-rome-round-logo.png"
)

# DataFrame
df = pd.read_csv('diabetes.csv')

# HEADINGS
image =  Image.open('ssm.png')
st.image(image,use_column_width=True)
st.write("""
The application can detect whether a person has diabetes or not based on a classification algorithm.
""")

# Patient Data (Sidebar)
st.sidebar.header('Patient Data')

# H2 Explore the data
st.subheader('Explore the data')
st.write(df.describe())

# FUNCTION
def user_report():
    Pregnancies = st.sidebar.slider('Pregnancies', 0,17, 3 )
    Glucose = st.sidebar.slider('Glucose', 0,200, 120 )
    BloodPressure = st.sidebar.slider('Blood Pressure', 0,122, 70 )
    SkinThickness = st.sidebar.slider('Skin Thickness', 0,100, 20 )
    Insulin = st.sidebar.slider('Insulin', 0,846, 79 )
    BMI = st.sidebar.slider('BMI', 0,67, 20 )
    DiabetesPedigreeFunction = st.sidebar.slider('Diabetes Pedigree Function', 0.0,2.4, 0.47 )
    Age = st.sidebar.slider('Age', 21,88, 33 )

    user_report_data = {
        'Pregnancies': Pregnancies,
        'Glucose': Glucose,
        'BloodPressure': BloodPressure,
        'SkinThickness': SkinThickness,
        'Insulin': Insulin,
        'BMI': BMI,
        'DiabetesPedigreeFunction': DiabetesPedigreeFunction,
        'Age': Age
    }
    report_data = pd.DataFrame(user_report_data, index=[0])
    return report_data


# PATIENT DATA
user_data = user_report()
st.subheader('Patient Data')
st.write(user_data)

#Model
x = df.drop(['Outcome'], axis = 1)
y = df.iloc[:, -1]
x_train, x_test, y_train, y_test = train_test_split(x,y, test_size = 0.2, random_state = 0)
Diabete = pickle.load(open('diabete.pkl', 'rb'))
user_result = Diabete.predict(user_data)

# OUTPUT
st.subheader('Your Report: ')
output = ''
if user_result[0] == 0:
    output = 'You are probably not Diabetic'
else:
    output = 'You are probably Diabetic, please consult your medical doctor'
st.title(output)

st.subheader('Accuracy: ')
st.write('The Precision of the model is ' + (str(accuracy_score(y_test, Diabete.predict(x_test)) * 100)) + '%')

# VISUALISATIONS
st.subheader('Visualised Patient Report')
st.bar_chart(df)

# COLOR FUNCTION
if user_result[0] == 0:
    color = 'green'
else:
    color = 'red'

# Comparing the data
st.header('Compare your Data')

# Age vs Pregnancies
st.subheader('Pregnancy count Graph (Others vs Yours)')
fig_preg = plt.figure()
ax1 = sns.scatterplot(x='Age', y='Pregnancies', data=df, hue='Outcome', palette='Greens')
ax2 = sns.scatterplot(x=user_data['Age'], y=user_data['Pregnancies'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 20, 2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_preg)

# Age vs Glucose
st.subheader('Glucose Value Graph (Others vs Yours)')
fig_glucose = plt.figure()
ax3 = sns.scatterplot(x='Age', y='Glucose', data=df, hue='Outcome', palette='magma')
ax4 = sns.scatterplot(x=user_data['Age'], y=user_data['Glucose'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 220, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_glucose)

# Age vs Bp
st.subheader('Blood Pressure Value Graph (Others vs Yours)')
fig_bp = plt.figure()
ax5 = sns.scatterplot(x='Age', y='BloodPressure', data=df, hue='Outcome', palette='Reds')
ax6 = sns.scatterplot(x=user_data['Age'], y=user_data['BloodPressure'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 130, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bp)

# Age vs St
st.subheader('Skin Thickness Value Graph (Others vs Yours)')
fig_st = plt.figure()
ax7 = sns.scatterplot(x='Age', y='SkinThickness', data=df, hue='Outcome', palette='Blues')
ax8 = sns.scatterplot(x=user_data['Age'], y=user_data['SkinThickness'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 110, 10))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_st)

# Age vs Insulin
st.subheader('Insulin Value Graph (Others vs Yours)')
fig_i = plt.figure()
ax9 = sns.scatterplot(x='Age', y='Insulin', data=df, hue='Outcome', palette='rocket')
ax10 = sns.scatterplot(x=user_data['Age'], y=user_data['Insulin'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 900, 50))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_i)

# Age vs BMI
st.subheader('BMI Value Graph (Others vs Yours)')
fig_bmi = plt.figure()
ax11 = sns.scatterplot(x='Age', y='BMI', data=df, hue='Outcome', palette='rainbow')
ax12 = sns.scatterplot(x=user_data['Age'], y=user_data['BMI'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 70, 5))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_bmi)

# Age vs Dpf
st.subheader('DPF Value Graph (Others vs Yours)')
fig_dpf = plt.figure()
ax13 = sns.scatterplot(x='Age', y='DiabetesPedigreeFunction', data=df, hue='Outcome', palette='YlOrBr')
ax14 = sns.scatterplot(x=user_data['Age'], y=user_data['DiabetesPedigreeFunction'], s=150, color=color)
plt.xticks(np.arange(10, 100, 5))
plt.yticks(np.arange(0, 3, 0.2))
plt.title('0 - Healthy & 1 - Unhealthy')
st.pyplot(fig_dpf)

#Hide streamlit header and footer

hide_st_style = """
            <style>
            #MainMenu {visibility: hidden;}
            footer {visibility: hidden;}
            header {visibility: hidden;}
            </style>
            """
st.markdown(hide_st_style, unsafe_allow_html=True)
