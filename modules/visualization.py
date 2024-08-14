import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd


def df_chart_display_loc(df: pd.DataFrame, data_col_loc: str):
    st.write(df.loc[:])
    st.line_chart(df.loc[:, data_col_loc])


def df_chart_display_iloc(df: pd.DataFrame, data_col_iloc: int):
    st.write(df.iloc[:])
    st.line_chart(df.iloc[:, data_col_iloc])


def plot_statistical_analysis(analysis_results):
    fig_acf_pacf, (ax1, ax2) = plt.subplots(2, 1, figsize=(10, 8))
    ax1.plot(analysis_results["acf"])
    ax1.set_title("Автокорреляция")
    ax2.plot(analysis_results["pacf"])
    ax2.set_title("Частичная автокорреляция")

    return fig_acf_pacf


def plot_decomposition(decomposition):
    fig, (ax1, ax2, ax3, ax4) = plt.subplots(4, 1, figsize=(10, 12))
    decomposition.observed.plot(ax=ax1)
    ax1.set_title("Исходный ряд")
    decomposition.trend.plot(ax=ax2)
    ax2.set_title("Тренд")
    decomposition.seasonal.plot(ax=ax3)
    ax3.set_title("Сезонность")
    decomposition.resid.plot(ax=ax4)
    ax4.set_title("Остаток")
    plt.tight_layout()
    return fig
