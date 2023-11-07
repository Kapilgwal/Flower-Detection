import streamlit as st
import pickle
import numpy as np
import re

pipe = pickle.load(open('flower.pkl','rb'))
df = pickle.load(open('df.pkl','rb'))

st.title("Flower Prediction")

# sepal length
l1 = st.number_input("Sepal Length of the flower")
# sepal widht
l2 = st.number_input("Sepal Widht of the flower")
# petal length
l3 = st.number_input("Petal Length of the flower")
# petal width
l4 = st.number_input("Petal Width of the flower")

if st.button('Prediction'):
    query = np.array([l1,l2,l3,l4],dtype = object)

    query = query.reshape(1,4)
    text = str(int(np.exp(pipe.predict(query)[0])))
    result = ""
    if text == 0:
        result = 'Iris-setosa'
    if text == 1:
        result ='Iris-versicolor'
    else:
        result ='Iris-virginica'


    st.title("The predicted flower of this length and width is " + result)
