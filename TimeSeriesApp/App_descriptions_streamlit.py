import streamlit as st
import pandas as pd


def sample_csv_download_button():
    with st.expander("Что делать, если нет своего ряда?"):

        def convert_df(df):
            # IMPORTANT: Cache the conversion to prevent computation on every rerun
            return df.to_csv().encode("utf-8")

        convert_df(pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/TimeSeriesApp/TimeSeriesApp/Vietnam_CO2_Temp.csv").iloc[:, 2:])
        st.write("Настоящий ряд для тестирования можно скачать тут")
        st.download_button(
            label="Download data as CSV",
            data=convert_df(pd.read_csv("/Users/arzangulyan/Documents/Научка/Vesna2022/Programming/TimeSeriesApp/TimeSeriesApp/Vietnam_CO2_Temp.csv")),
            file_name="Vietnam.csv",
            # mime='text/csv',
        )


def intro_text():
    with st.expander("О комплексе?"):
        st.write(
            "Комплекс разработан для работы с временными рядами метеорологических данных. Он позволяет совершить предобаботку ряда, а также его анализ и прогнозирование используя один из представленных в боковом меню методов. Комплекс доступен для запуска с любого устройства онлайн по ссылке https://time-series-msu-ff.streamlit.app/"
        )


def preprocessing():
    st.write(
        "На этапе предобработки ряда мы можем выявить пропуски в данных, проверить ряд на стационарность и провести его сглаживание"
    )


def stationar_test():
    st.write(
        "Проводится стандартный тест Дики-Фуллера (DF-тест), который является статистическим методом, применяемым для проверки стационарности временных рядов. Он используется для определения наличия единичного корня в ряде, что указывает на его нестационарность. \n Получаемый параметр p-value является уровнем значимости гипотезы. Если он не выходит за пределы 5%, то мы считаем наш ряд стационарным."
    )

def moving_average():
    st.write("Сглаживание позволяет устранить шумы и случайные колебания. Оно реализуется методом скользящего среднего. Его суть заключается в преобразовании исходного ряда в новый, где каждое значение является средним арифметическим значений в окне заданной ширины")  
    st.latex(r''' MA(t) = \frac{1}{2n+1} \sum_{i=t-n}^{t+n} y_i''')