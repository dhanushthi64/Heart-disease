import numpy as np
import pickle
import streamlit as st
# loading the classifier
loadmodel = pickle.load(open(
    'C:/Users/User/Desktop/Heart Disease Prediction/notebooks/trained_heart_model.sav', 'rb'))
# function for prediction


def heart_disease_pred(input_data):
    numpy_array = np.asarray(input_data)
    input_data_reshape = numpy_array.reshape(1, -1)
    prediction = loadmodel.predict(input_data_reshape)
    print(prediction)
    if (prediction[0] == 0):
        return "The person does not have heart attack"
    else:
        return "The person have heart attack"

# Main function


def main():
    # giving title
    st.title('Heart Disease Prediction')
    # getting input data from the user
    # age,sex,cp,trestbps,chol,fbs,restecg,thalach,exang,oldpeak,slope,ca,thal
    age = st.text_input('Age:')
    sex = st.text_input('Sex')
    cp = st.text_input('Chest Pain Type:')
    trestbps = st.text_input('Resting Blood Pressure:')
    chol = st.text_input('Cholestrol:')
    fbs = st.text_input('Fasting Glucose Level:')
    restecg = st.text_input('ElectroCardiographic Measurement:')
    thalach = st.text_input('Maximum Heart Rate:')
    exang = st.text_input('Exercise Induced Angina:')
    oldpeak = st.text_input('Depression:')
    slope = st.text_input('Heart Rate Slope:')
    ca = st.text_input('Calcium Rate:')
    thal = st.text_input('Iron Rate:')
    # code for prediction
    diagnosis = ''
    # prediction button
    if st.button('Predict'):
        diagnosis = heart_disease_pred(
            [age, sex, cp, trestbps, chol, fbs, restecg, thalach, exang, oldpeak, slope, ca, thal])
    st.success(diagnosis)


if __name__ == '__main__':
    main()
