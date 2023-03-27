import numpy as np
import matplotlib.pyplot as plt
import pywt
import seaborn as sns
import pandas as pd
from statsmodels.tsa.arima_model import ARMA
from statsmodels.tsa.stattools import adfuller
from statsmodels.tsa.ar_model import AutoReg




# from statsmodels.tsa.stattools import adfuller


# x = '1/1/12 01:30'
# month = ''
# print(type(x))
# i = ''
# while i != '/':
#     for i in x:
#         month += i
#         print (month)


    # while i != '/':

    #     print (month)
# X = np.arange(512)
# Y = np.sin(2*np.pi*X/5)
# coef, freq = pywt.cwt(Y, np.arange(1, 75),'gaus1')
# coef
# plt.matshow(coef)
# plt.show()

def plot_wavelet(time, signal, scales, waveletname = 'mexh', cmap = plt.cm.seismic, title = 'Wavelet Transform (Power Spectrum) of signal', ylabel = 'Period (days)', xlabel = 'Time'):
    dt = time[1] - time[0]
    [coefficients, frequencies] = pywt.cwt(signal, scales, waveletname, dt)
    global power
    global period
    power = (abs(coefficients))
    period = (1. / frequencies)*365.25
    contourlevels = np.linspace(power.min(), power.max(), 5)
    contourlevels
    coefficients.mean()
    coefficients.min()
    coefficients.max()
    power.max()


    fig, ax = plt.subplots(figsize=(15, 7))
    im = ax.contourf(time, np.log2(period), power, contourlevels, extend='both',cmap=cmap)

    ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    yticks = 2**np.arange(np.ceil(np.log2(period.min())), np.ceil(np.log2(period.max())))
    ax.set_yticks(np.log2(yticks))
    ax.set_yticklabels(yticks)
    ax.invert_yaxis()
    ylim = ax.get_ylim()
    ax.set_ylim(bottom=ylim[0])

    cbar_ax = fig.add_axes([0.9, 0.5, 0.03, 0.25])
    fig.colorbar(im, cax=cbar_ax, orientation="vertical")
    plt.show()



x = np.arange(1000)
# y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/700)
y = pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/Attempt_1/Vietnam_CO2_Temp.csv")
y = y.loc[:10000, 'Temperature']
df = y
N = y.size
t0 = 2012
dt = 5.7*10**(-5)
time = np.arange(0, N) * dt + t0
signal = y
scales = np.arange(1, N/4)


signal.plot(figsize=(10,6))
# signal.rolling(window=20).mean().plot()
# df['SMA10'] =
# print(df)
print(df.info)

test_res = adfuller(signal)
print(test_res[1], end='\n')
# print(test_res, end='\n')
# print(test_res[:3], end='\n')
# print(test_res, end='\n')

from statsmodels.graphics.tsaplots import plot_acf, plot_pacf
pacf = plot_pacf(signal, lags=25)
acf = plot_acf(signal, lags=25)
plt.show()


# plot_wavelet(time, signal, scales)

#
#
#
#
#
#
# x = np.arange(1000)
# y = np.sin(2*np.pi*x/100) + np.cos(2*np.pi*x/700)
# coef, freqs=pywt.cwt(y,np.arange(1,128),'mexh')
# freqs.shape
# coef.shape
# # fig1, ax1 = plt.subplots()
# plt.plot(x,y)
# plt.figure()
# plt.imshow(coef, cmap = 'copper', aspect = 'auto')
#
# # # plt.matshow(coef) # doctest: +SKIP
# plt.xlabel('time')
# plt.ylabel('frequency')
# plt.figure()

#
#

# range(len(nfreqs)-1)
# power.shape
def Intergral_spectrum (time, period, power, xlabel="Period", ylabel="Power Spectrum"):
    I = np.empty((len(period), 1))
    for j in range(len(period)-1):
        for i in range(N):
            # I[j] += ((power[j, i])**2 + (power[j+1,i])**2)/2;
            I[j] += ((power[j, i]) + (power[j+1,i]))/2;
    fig, ax = plt.subplots(figsize=(15, 7))
    ax.plot(np.log2(period), np.log2(I))

    # ax.set_title(title, fontsize=20)
    ax.set_ylabel(ylabel, fontsize=18)
    ax.set_xlabel(xlabel, fontsize=18)

    # xticks = 2**np.arange(np.ceil(period.min()), np.ceil(period.max()))
    # ax.set_xticks(np.log2(xticks))
    # ax.set_xticklabels(xticks)
    # ylim = ax.get_ylim()
    ax.set_ylim(auto=True)
    plt.show()



# Intergral_spectrum(time, period, power)




# freqs.shape
# nfreqs.shape
# plt.plot(nfreqs, I)
#
# #
# plt.show()
