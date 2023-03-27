import streamlit as st

st.title("My New Project")
st.write("Oh no, I'm gonna type smth here")
button1 = st.button("Click me, pls")

if button1:
    st.write("Hooray!")

st.header("Checkbox section")
like = st.checkbox("Do you like App?")
button2 = st.button("Submit")
if button2:
    if like:
        st.write("thx, i like it 2")
    else:
        st.write("u have bad taste")

st.header("NEW CHAPTER")

animal = st.radio("what is your fav animal?", ("lions", "wolves", "rats"))

button3 = st.button("Submit animal")
if button3:
    if animal == "lions":
        st.write("rooooar")


st.header("VERY NEW CHAPTER")

animal2 = st.selectbox("what is your fav animal?", ("lions", "wolves", "rats"))

button4 = st.button("Submit animal2")
if button4:
    if animal2 == "lions":
        st.write("rooooar")

st.header("Start of multiselect")
options = st.multiselect("What an-s do u like", ("lions", "tigers", "wolves"))
button5 = st.button("chosen an-s")
if button5:
    st.write(options)

st.header("Slider section")
epochs_num = st.slider("How many epochs?", 1,100, 10, 5)
if st.button("Slider button"):
    st.write(epochs_num)

st.header("Text input section")

user_txt = st.text_input("Whats ur fav movie", "Star Wars", )
if st.button("Movie name"):
    st.write(user_txt)

user_num = st.number_input("whats ur fav number")
if st.button("Number button"):
    st.write(user_num)
