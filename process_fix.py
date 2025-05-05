import streamlit as st
import pandas as pd 
import numpy as np
import traceback
import datetime

def process_time_series(time_series):
    """Обработка временного ряда"""
    # Инициализируем текущее состояние временного ряда
    if "current_time_series" not in st.session_state:
        st.session_state.current_time_series = None
        
    try:
        # Логируем начальное состояние
        with st.expander("🔍 Отладочная информация", expanded=False):
            st.subheader("Отладочная информация о временном ряде")
            log_dataframe_state(time_series, "Исходный временной ряд", detailed=True)
        
        # Отображение информации о датасете
        st.header("Общая информация о данных")
        # ...остальной код...
        
        # Шаг 5: Обработка пропущенных значений
        try:
            if time_series_selected.isna().any().any():
                st.subheader("5. Обработка пропущенных значений")
                # Код обработки пропущенных значений...
        except Exception as e:
            st.error(f"Ошибка при обработке пропущенных значений: {str(e)}")
            st.error(f"Трассировка: {traceback.format_exc()}")
        
        # Завершение функции
        st.success(f"Обработка завершена. Размер итогового набора данных: {time_series_selected.shape[0]} записей, {time_series_selected.shape[1]} колонок")
        
        # Обновляем состояние processed_data в session_state
        st.session_state.processed_data = time_series_selected.copy(deep=True)
        
        # Финальное обновление текущего состояния
        if "update_current_time_series" in globals():
            time_series_selected = update_current_time_series(time_series_selected, "Завершение обработки")
        
        return time_series_selected
        
    except Exception as e:
        st.error(f"Ошибка при обработке временного ряда: {str(e)}")
        st.error(f"Тип ошибки: {type(e)}")
        st.error(f"Трассировка: {traceback.format_exc()}")
        return None 

def safely_process_missing_values(time_series):
    """
    Безопасно обрабатывает пропущенные значения в DataFrame.
    
    Args:
        time_series: DataFrame с временным рядом
        
    Returns:
        Обработанный DataFrame
    """
    try:
        if time_series.isna().any().any():
            st.subheader("5. Обработка пропущенных значений")
            
            # Показываем информацию о пропущенных значениях
            missing_data = time_series.isna().sum()
            st.write("Количество пропущенных значений по колонкам:")
            st.dataframe(missing_data.to_frame(name="Пропущенные значения"), use_container_width=True)
            
            na_method = st.radio(
                "Выберите метод обработки пропущенных значений",
                options=["Оставить как есть", "Удалить строки", "Заполнить нулями", "Заполнить средним", "Интерполировать"]
            )
            
            if na_method == "Удалить строки":
                orig_shape = time_series.shape[0]
                time_series = time_series.dropna()
                st.success(f"Удалено {orig_shape - time_series.shape[0]} строк с пропущенными значениями")
                
                # Обновляем текущее состояние, если функция существует
                if "update_current_time_series" in globals():
                    time_series = update_current_time_series(time_series, "Удаление пропущенных значений")
                
            elif na_method == "Заполнить нулями":
                time_series = time_series.fillna(0)
                st.success("Пропущенные значения заполнены нулями")
                
                # Обновляем текущее состояние, если функция существует
                if "update_current_time_series" in globals():
                    time_series = update_current_time_series(time_series, "Заполнение нулями")
                
            elif na_method == "Заполнить средним":
                time_series = time_series.fillna(time_series.mean())
                st.success("Пропущенные значения заполнены средними значениями")
                
                # Обновляем текущее состояние, если функция существует
                if "update_current_time_series" in globals():
                    time_series = update_current_time_series(time_series, "Заполнение средними значениями")
                
            elif na_method == "Интерполировать":
                time_series = time_series.interpolate(method='time')
                st.success("Пропущенные значения заполнены с помощью интерполяции")
                
                # Обновляем текущее состояние, если функция существует
                if "update_current_time_series" in globals():
                    time_series = update_current_time_series(time_series, "Интерполяция пропущенных значений")
        
        return time_series
    except Exception as e:
        st.error(f"Ошибка при обработке пропущенных значений: {str(e)}")
        st.error(f"Трассировка: {traceback.format_exc()}")
        return time_series  # Возвращаем исходный ряд в случае ошибки 