import numpy as np
import pickle
#Loading The Model
loadmodel=pickle.load(open('C:/Users/User/Desktop/Heart Disease Prediction/notebooks/trained_heart_model.sav','rb'))
#testing
input_data=(53,1,0,140,203,1,0,155,1,3.1,0,0,3)
numpy_array=np.asarray(input_data)
input_data_reshape=numpy_array.reshape(1,-1)
prediction=loadmodel.predict(input_data_reshape)
print(prediction)
if (prediction[0]==0):
    print("The person does not have heart attack")
else:
    print("The person have heart attack")