import pickle as pkl
import pandas as pd
import numpy as np
import streamlit as st
def data_pipepline(input_copy):
    crt_ft = ['Geography', 'Gender']
    with open('models/encoder.pkl','rb') as file:
        encoder=pkl.load(file)
    with open('models/scaler.pkl','rb') as file:
        scaler=pkl.load(file)
    with open('models/rf_model.pkl','rb') as file:
        rf_model=pkl.load(file)
    encoded_features = encoder.transform(input_copy[crt_ft])
    encoded_df = pd.DataFrame(encoded_features, columns=encoder.get_feature_names_out(crt_ft))
    input_copy= pd.concat([input_copy, encoded_df], axis=1)
    input_copy.drop(crt_ft, axis=1, inplace=True)
    input_copy = scaler.transform(input_copy)
    ans = rf_model.predict(input_copy)
    # Print predictions
    return ans
    # print(f'Prediction from best_rf_model: {ans}')
st.title('Bank Customer Churn Prediction')
st.markdown("""
        This model is built with using Random-forrest Classifier from scikit-learn library with an overall accuracy of 85%, effectively distinguishing between customers likely to stay and those at risk of leaving.
            
**Precision and Recall**: The model achieved a precision of 88% and a recall of 95% for predicting customers who are not likely to churn, reflecting its reliability in identifying loyal customers. For those likely to churn, the model has a precision of 70% and a recall of 48%, highlighting the challenges in predicting churn but still providing valuable insights.

**F1-Score**: With an F1-score of 91% for non-churning customers and 57% for those at risk, the model balances precision and recall, ensuring comprehensive churn prediction.

""")
st.subheader('Confusion Matrix:')
st.image('images/confusion_matrix.png')
# data=pd.read_csv('data.csv')
# data_pipepline(data)
CreditScore=st.text_input('CreditScore')
Geography=st.selectbox('Geography',['France','Germany','Spain'])
Age=st.text_input('Age')
Tenure=st.text_input('Tenure')
Balance=st.text_input('Balance')
NumOfProducts=st.text_input('Number of Products used')
HasCrCard=st.selectbox('Has a Credit card',['Yes','No'])
IsActiveMember=st.selectbox('Is an active member',['Yes','No'])
EstimatedSalary=st.text_input('Estimated Salary of the Customer')
Gender=st.selectbox('Gender',['Male','Female'])
HasCrCard = 1 if HasCrCard == 'Yes' else 0
IsActiveMember = 1 if IsActiveMember == 'Yes' else 0
def convert_input(input_value, data_type, default_value):
    if input_value.strip() == '':
        return default_value
    else:
        return data_type(input_value)
data = {
    'CreditScore': [convert_input(CreditScore, int, 0)],
    'Geography': [Geography],
    'Gender':[Gender],
    'Age': [convert_input(Age, int, 0)],
    'Tenure': [convert_input(Tenure, int, 0)],
    'Balance': [convert_input(Balance, float, 0.0)],
    'NumOfProducts': [convert_input(NumOfProducts, int, 0)],
    'HasCrCard': [HasCrCard],
    'IsActiveMember': [IsActiveMember],
    'EstimatedSalary': [convert_input(EstimatedSalary, float, 0.0)]
}
input_df = pd.DataFrame(data)
if(st.button('Predict')):
    res=data_pipepline(input_df)
    if(res):
        st.subheader('Customer at risk of leaving')
    else:
        st.subheader('Customer likely to stay')
