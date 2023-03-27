import streamlit as st

st.title("Layouts and images")

st.sidebar.header("This is options menu")
txt = st.sidebar.text_area("Paste txt")
button1 = st.sidebar.button("clean txt")
if button1:
    col1, col2 = st.columns(2)
    c1_exp = col1.expander("Expand #1")
    with c1_exp:
        c1_exp.header("Column #1")
        c1_exp.write(txt)
    c2_exp = col2.expander("Expand #2")
    with c2_exp:
        c2_exp.header("Column #2")
        c2_exp.write(txt)
