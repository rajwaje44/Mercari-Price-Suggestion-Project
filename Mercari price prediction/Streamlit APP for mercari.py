

import streamlit as st
import numpy as np
import joblib
import pickle
import pandas as pd
from PIL import Image


#ValueError: If using all scalar values, you must pass an index
#Above error occurs if object is not in a list format

# image
img = Image.open("D:/python pycharm/Imarticus Project/RORARA circle.png")
st.image(img,width=175)

#st.title("Mercari Price Prediction")
html_temp = """
    <div style="background-color:orange;padding:0px">
    <h1 style="color:black;text-align:center;">Mercari Price Prediction </h1>
    </div>
    """
st.markdown(html_temp,unsafe_allow_html=True)
st.subheader("Fill below fields to predict the Price")

item_id = [st.selectbox("Item Condition ID",["1","2","3","4","5"])]
shi = [st.selectbox("Shipping",["0","1"])]
bn = [st.text_input("Enter Brand Name","Type here")]
mc = [st.selectbox("Enter Main Category",["Women","Beauty","Kids","Electronics","Men","Home","Other","Vintage & Collectibles","Handmade","Sports & Outdoors"])]
sc1 = [st.text_input("Enter Sub-Category1","Type here")]
sc2 = [st.text_input("Enter Sub-Category2","Type here")]

if st.button("Predict"):
    ds = {'item_condition_id': item_id, 'shipping': shi}

    with open("D:/python pycharm/New folder/NLP/nn", "rb") as f:
        nnn = pickle.load(f)

    nndff = pd.DataFrame(data=ds, columns=nnn)
    nndff[bn] = 1
    nndff[mc] = 1
    nndff[sc1] = 1
    nndff[sc2] = 1
    nndff.fillna(value=0, inplace=True)

    rf_model = joblib.load("D:/python pycharm/Imarticus Project/model_rf.pkl")
    predictor = rf_model
    n = predictor.predict(nndff)
    value = np.exp(n)
    value = "Price is $"+str(float(value))
    st.success(value)






