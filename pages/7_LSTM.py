import streamlit as st
import numpy as np
import pandas as pd
import App_descriptions_streamlit as txt
from modules.lstm_module import LSTM_ts, calculate_metrics
from modules.utils import nothing_selected
from modules.page_template import setup_page, load_time_series, display_data
import warnings

warnings.filterwarnings("ignore", category=UserWarning, module="keras")


def main():
    setup_page("LSTM", "Настройки модели LSTM")
    txt.LSTM_descr()

    time_series = load_time_series()

    epochs = int(st.sidebar.number_input("Количество эпох", min_value=1, value=10))
    forecast_periods = int(
        st.sidebar.number_input(
            "Количество периодов для прогноза", min_value=1, value=1
        )
    )
    txt.LSTM_epochs_choice()

    signal = time_series.iloc[:, 0]
    data, train_plot, test_plot, future_plot = LSTM_ts(signal, epochs, forecast_periods)

    results_df = pd.DataFrame(
        {
            "Date": pd.date_range(start=time_series.index[0], periods=len(data)),
            "Original Data": data,
            "Train Predict": train_plot,
            "Test Predict": test_plot,
            "Future Predict": future_plot,
        }
    )
    results_df.set_index("Date", inplace=True)

    st.subheader("LSTM Predictions and Forecast")
    st.line_chart(results_df)

    train_rmse = calculate_metrics(
        results_df["Original Data"][results_df["Train Predict"].notna()],
        results_df["Train Predict"].dropna(),
    )
    test_rmse = calculate_metrics(
        results_df["Original Data"][results_df["Test Predict"].notna()],
        results_df["Test Predict"].dropna(),
    )
    st.write(f"Train RMSE: {train_rmse:.4f}")
    st.write(f"Test RMSE: {test_rmse:.4f}")

    # Отображение прогнозов на будущие периоды
    future_predictions = results_df["Future Predict"].dropna()
    if not future_predictions.empty:
        st.write("Прогноз на будущие периоды:")
        st.dataframe(future_predictions)
    else:
        st.write("Нет прогнозов на будущие периоды.")


if __name__ == "__main__":
    main()
