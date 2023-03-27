import streamlit as st
import numpy as np
import pandas as pd
import pywt
import matplotlib.pyplot as plt
import seaborn as sns
import altair as alt
from numpy.fft import fft
from statsmodels.tsa.stattools import adfuller

st.set_page_config(page_title="Wavelets")

st.markdown("# Wavelets Demo")
st.sidebar.header("Wavelets Demo")