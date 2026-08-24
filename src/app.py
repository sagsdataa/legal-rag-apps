import streamlit as st
import os

st.write("Current Directory:")
st.write(os.getcwd())

st.write("Files in Current Directory:")
st.write(os.listdir("."))

st.write("Parent Directory:")
st.write(os.listdir(".."))
