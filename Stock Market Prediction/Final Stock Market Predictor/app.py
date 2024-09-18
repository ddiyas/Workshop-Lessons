import pandas as pd
import streamlit as st
import yfinance as yf

sp500 = yf.Ticker("^GSPC")
sp500 = sp500.history(period="max")

sp500_for_chart = sp500[["Close"]]
sp500_for_chart = sp500_for_chart.set_index(pd.to_datetime(sp500.index))

st.title("Stock Market Prediction App")
st.line_chart(sp500_for_chart)

st.write("Should you invest today?")

prediction = 9
with open("tomorrow_prediction.txt", "r") as file:
    prediction = file.read()

st.write('No, today is not the best day to invest' if prediction == "0.0" else 'Yes, today is a good day to invest!')