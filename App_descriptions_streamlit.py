import streamlit as st
import pandas as pd





def intro_text():
    with st.expander("О комплексе?"):
        st.write(
            "Комплекс разработан для работы с временными рядами метеорологических данных. Он позволяет совершить предобаботку ряда, а также его анализ и прогнозирование используя один из представленных в боковом меню методов. Комплекс доступен для запуска с любого устройства онлайн по ссылке https://time-series-phys-msu.streamlit.app"
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

def LSTM_descr():
    with st.expander("Что такое метод LSTM?"):
        st.write(
            "**Long Short-Term Memory (LSTM)** — это тип рекуррентной нейронной сети (RNN), специально разработанный для эффективного анализа и предсказания временных рядов. Традиционные RNN сталкиваются с проблемой кратковременной памяти — они не могут эффективно запоминать долговременные зависимости ввиду их затухания с течением времени. LSTM решает эту проблему с помощью своей уникальной архитектуры, включающей в себя «ячейки памяти» и управляемые «вратами» (входными, забывающими и выходными), которые могут добавлять или удалять информацию из ячеек памяти в зависимости от необходимости.")
            
        st.write(
            "Эта структура позволяет LSTM запоминать важные паттерны на длительных временных интервалах, делая их особенно полезными в таких приложениях, как предсказание временных рядов. Благодаря LSTM можно более точно моделировать сложные временные зависимости и улучшать прогнозирование трендов или аномалий по сравнению с традиционными методами анализа временных рядов.")

def LSTM_epochs_choice():
    with st.sidebar.expander("Как выбрать кол-во эпох?"):
        st.write(
            "**Эпоха** — это один полный проход через весь обучающий набор данных. Увеличение количества эпох может улучшить качество модели, но слишком большое количество эпох может привести к переобучению, когда модель будет хорошо работать на обучающих данных, но плохо на тестовых.")
            
        st.write(
            "Для начала рекомендуется установить небольшое количество эпох (до  5) и постепенно увеличивать его, отслеживая при этом качество прогноза на валидационном наборе данных. Оптимальное количество эпох достигается, когда модель показывает наилучшие результаты на валидационных данных без признаков переобучения.")
        
def ARMA_descr():
    with st.expander("Что такое метод ARMA?"):
        st.write(
            "ARMA (Autoregressive Moving Average) — это метод анализа и предсказания временных рядов, который сочетает в себе два компонента: авторегрессионный (AR) и скользящего среднего (MA). Авторегрессионная часть модели описывает текущее значение временного ряда как линейную комбинацию нескольких его предыдущих значений, а часть скользящего среднего моделирует текущее значение как комбинацию прошлых ошибок предсказания. Это позволяет ARMA моделям эффективно захватывать как тренды, так и зависимости в данных временных рядов."
        )

def ARIMA_descr():
    with st.expander("Что такое метод ARIMA?"):
        st.write(
            "ARIMA (Autoregressive Integrated Moving Average) — это расширение модели ARMA, которое включает компонент интегрированности (I). Компонент интегрированности позволяет модели работать с нестационарными временными рядами, путем дифференцирования данных для удаления трендов и сезонных зависимостей. Таким образом, ARIMA включает в себя авторегрессионную (AR), интегрированную (I) и скользящего среднего (MA) части, что делает её мощным инструментом для предсказания временных рядов с трендами и сезонными эффектами."
        )

def Wavelet_descr():
    with st.expander("Что такое Вейвлет преобразование?"):
        st.write(
            "Вейвлет-преобразование (Wavelet Transform) — это метод анализа временных рядов, который позволяет разложить сигнал на составляющие с различной частотой и разрешением. В отличие от традиционного Фурье-преобразования, которое разлагает сигнал на синусоиды, вейвлет-преобразование использует функции, называемые вейвлетами. Эти функции локализованы по времени и частоте, что позволяет более точно анализировать временные ряды с быстро меняющимися характеристиками. Вейвлет-преобразование широко применяется в задачах обнаружения аномалий, сжатия данных и анализа мульти-частотных сигналов."
        )

def Fourier_descr():
    with st.expander("Что такое Фурье преобразование?"):
        st.write(
            "Фурье-преобразование (Fourier Transform) — это метод преобразования временного ряда из временной области в частотную. Оно разлагает сигнал на набор синусоид с разными частотами, амплитудами и фазами. Это позволяет выявить частотные компоненты в данных и понять, какие частоты присутствуют в сигнале. Обратное Фурье-преобразование позволяет восстановить исходный сигнал из частотных компонент. Этот метод важен для анализа и фильтрации сигналов, обнаружения периодических элементов, а также для сжатия данных и восстановления сигналов."
        )

# def ARMA_params_choice():
#     with st.sidebar.expander("Как выбрать параметры?"):
#         st.write(
#             "Параметр p определяется ...")
            
#         st.write(
#             "Параметр q определяется ...")
        
def ARMA_params_choice():
    with st.sidebar.expander("Как выбрать параметры?"):
        
        st.write( 
            "Параметр p определяется по первым значимым лагам ЧАКФ. Если ЧАКФ резко падает и обнуляется после определенного лага, это является индикатором порядка AR, который подходит для моделирования временного ряда. Например, если ЧАКФ значительно обнуляется на четвертом лаге, это может указывать на AR(4) модель."
        )
        
        st.write(
            "Параметр q определяется по первым значимым лагам АКФ. Если АКФ показывает значительное убывание и становится незначимой после определенного лага, это знак подходящего порядка MA. Например, если АКФ обнуляется на втором лаге, это может указывать на MA(2) модель."
        )