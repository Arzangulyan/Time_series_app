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

def df_chart_display_iloc(df):
    col1, col2 = st.columns(2)
    with col1:
        st.write(df)
    with col2:
        st.line_chart(df.iloc[:, 1])

def wavelet_transform(time_series, mother_wavelet):
    mother_switcher = {
    "Морле": "morl",
    "Гаусс": "gaus1",
    "Мексиканская шляпа": "mexh",
    "Другой": ""
}
    
    coef, freqs = pywt.cwt(time_series.iloc[:, 1], np.arange(1, len(time_series)/4), mother_switcher.get(mother_wavelet))
    return coef, freqs 

time_series = st.session_state.final_dataframe

df_chart_display_iloc(time_series)
wavelet_select = st.sidebar.selectbox(label='Выберите материнский вейвлет', options=(["", "Морле", "Гаусс", "Мексиканская шляпа", "Другой"]))



# st.stop()
if wavelet_select == "":
    st.stop()

# #TEST
# x = np.linspace(0, 1000,1000)
# y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/300) + np.cos(2*np.pi*x/900)
# linechart = st.line_chart(y)
# coef, freqs = pywt.cwt(y, np.arange(1,129), 'mexh')
# fig, ax = plt.subplots()
# ax.imshow(coef, cmap = 'copper', aspect = 'auto')
# st.pyplot(fig)
# # sns.heatmap(coef, ax = ax, cmap = 'copper')
# # st.write(fig)
#
# I = np.empty((len(freqs)))
# for j in range(len(freqs)-1):
#     for i in range(len(y)):
#         I[j] += ((coef[j, i])**2 + (coef[j+1,i])**2)/2
# # st.write(I)
# I_s = pd.DataFrame({'I':I, 'Freqs': freqs})
# st.write(I_s)
# fig2, ax2 = plt.subplots()
# ax2.plot(freqs, I)
# ax2.set_aspect('auto')
# st.pyplot(fig2)
#
# # Intergral_spectrum = st.line_chart(I_s)
# #TEST

#REAL

coef, freq = wavelet_transform(time_series, wavelet_select)
fig1, ax1 = plt.subplots()
ax1.imshow(coef, cmap = 'copper', aspect = 'auto')
# sns.heatmap(coef, ax = ax, cmap = 'copper')
# st.write(fig)
ax1.set_title("Power Spectrum", fontsize=20)
ax1.set_ylabel("Период", fontsize=18)
ax1.set_xlabel("Время", fontsize=18)
ax1.invert_yaxis()
st.pyplot(fig1)



# I = np.empty((len(freqs)))
# for j in range(len(freqs)-1):
#     for i in range(len(df_selected.loc[(start_point+m_a_step):end_point, 'Averaged'])-1):
#         I[j] += ((coef[j, i]) + (coef[j+1,i]))/2
# # st.write(I)
# Int_freq = pd.DataFrame({'I':I, 'Freqs': freqs})
# st.write(Int_freq)
# fig2, ax2 = plt.subplots()
# ax2.plot(freqs, I)
# ax2.set_aspect('auto')
# plt.xscale("log")
# st.pyplot(fig2)
#REAL