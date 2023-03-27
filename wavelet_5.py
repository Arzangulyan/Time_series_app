import streamlit as st

st.title("Forms part")

form1 = st.form(key="options")

form1.write("Hello world")
button1 = form1.form_submit_button("click me pls")
