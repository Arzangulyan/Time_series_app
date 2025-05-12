import io
import base64
import tempfile
import streamlit as st
from typing import Dict, Any, Optional
import matplotlib.pyplot as plt

try:
    import markdown2
except ImportError:
    markdown2 = None
try:
    import pdfkit
except ImportError:
    pdfkit = None
try:
    from weasyprint import HTML
except ImportError:
    HTML = None


def save_plot_to_base64(fig, backend: str = 'matplotlib') -> str:
    """
    Сохраняет график (matplotlib или plotly) в base64-строку для вставки в markdown.
    Улучшенная версия с настройками для лучшего отображения в PDF.
    """
    buf = io.BytesIO()
    if backend == 'matplotlib':
        # Увеличиваем размер и DPI для лучшей четкости
        fig.set_size_inches(10, 6)
        
        # Улучшаем отображение легенды
        if fig.legends or any(ax.get_legend() for ax in fig.axes):
            # Если есть легенда, добавляем больше места
            fig.subplots_adjust(right=0.85)
            
        # Увеличиваем размер шрифта для лучшей читаемости в PDF
        for ax in fig.axes:
            for item in ([ax.title, ax.xaxis.label, ax.yaxis.label] +
                        ax.get_xticklabels() + ax.get_yticklabels()):
                item.set_fontsize(12)
                
            # Если есть легенда, увеличиваем шрифт
            if ax.get_legend():
                for text in ax.get_legend().get_texts():
                    text.set_fontsize(10)
        
        plt.tight_layout()
        fig.savefig(buf, format="png", bbox_inches='tight', dpi=150)
    elif backend == 'plotly':
        # Увеличиваем размер и разрешение для plotly
        fig.write_image(buf, format="png", width=1000, height=600, scale=2)
    else:
        raise ValueError("Unknown backend: only 'matplotlib' or 'plotly' supported")
    buf.seek(0)
    img_base64 = base64.b64encode(buf.read()).decode("utf-8")
    return img_base64


def generate_markdown_report(
    title: str,
    description: str,
    metrics_train: Dict[str, Any],
    metrics_test: Dict[str, Any],
    train_time: float,
    forecast_img_base64: str,
    loss_img_base64: str,
    params: Optional[Dict[str, Any]] = None,
    early_stopping: bool = False,
    early_stopping_epoch: Optional[int] = None
) -> str:
    """
    Генерирует markdown-отчет по эксперименту с YAML front matter для автоматизированного анализа.
    """
    import datetime
    import uuid
    # Формируем YAML front matter
    experiment_id = f"LSTM_{datetime.datetime.now().strftime('%Y-%m-%d_%H-%M-%S')}_{str(uuid.uuid4())[:8]}"
    yaml_lines = [
        '---',
        f'experiment: "{experiment_id}"',
        f'date: "{datetime.datetime.now().strftime('%Y-%m-%d %H:%M:%S')}"',
        f'model: "LSTM"',
    ]
    if params:
        for k, v in params.items():
            yaml_lines.append(f'{k.replace(" ", "_").lower()}: {v}')
    yaml_lines.append(f'early_stopping: {str(early_stopping).lower()}')
    yaml_lines.append(f'early_stopping_epoch: {early_stopping_epoch if early_stopping_epoch is not None else "null"}')
    yaml_lines.append('train_metrics:')
    for k, v in metrics_train.items():
        yaml_lines.append(f'  {k}: {v}')
    yaml_lines.append('test_metrics:')
    for k, v in metrics_test.items():
        yaml_lines.append(f'  {k}: {v}')
    yaml_lines.append(f'train_time: {train_time:.2f}')
    yaml_lines.append('---\n')
    yaml_block = '\n'.join(yaml_lines)
    css = """<style>
    img { 
        display: block; 
        margin: 20px auto; 
        max-width: 100%; 
        height: auto; 
        box-shadow: 0 2px 5px rgba(0,0,0,0.1);
        border-radius: 4px;
    }
    @media print {
        img {
            max-width: 100%;
            page-break-inside: avoid;
        }
        h2 { 
            page-break-before: always; 
        }
        h2:first-of-type { 
            page-break-before: avoid; 
        }
    }
    </style>"""
    params_md = ""
    if params:
        params_md = '\n'.join([f"- **{k}**: {v}" for k, v in params.items()])
    early_stopping_md = ""
    if early_stopping:
        early_stopping_md = f"\n**Ранняя остановка:** Да, на эпохе {early_stopping_epoch}\n"
    else:
        early_stopping_md = "\n**Ранняя остановка:** Нет\n"
    
    # Safely format metrics with defaults for missing values
    def format_metric(metrics, key, format_str=".4f"):
        if key in metrics:
            return f"{metrics[key]:{format_str}}"
        return "Н/Д"  # Not available
    
    md = f"""{yaml_block}{css}
# {title}

{description}

## Параметры эксперимента
{params_md}
{early_stopping_md}
## Основные метрики

**Обучающая выборка:**
- RMSE: {format_metric(metrics_train, 'rmse')}
- MAE: {format_metric(metrics_train, 'mae')}
- MAPE: {format_metric(metrics_train, 'mape')}
- SMAPE: {format_metric(metrics_train, 'smape')}
- Theil's U2: {format_metric(metrics_train, 'theil_u2')}
"""

    # Add LSTM-specific metrics if they exist
    if 'mase' in metrics_train or 'r2' in metrics_train:
        md += f"""
- MASE: {format_metric(metrics_train, 'mase')}
- R²: {format_metric(metrics_train, 'r2')}
- Adjusted R²: {format_metric(metrics_train, 'adj_r2')}
"""

    md += f"""
**Тестовая выборка:**
- RMSE: {format_metric(metrics_test, 'rmse')}
- MAE: {format_metric(metrics_test, 'mae')}
- MAPE: {format_metric(metrics_test, 'mape')}
- SMAPE: {format_metric(metrics_test, 'smape')}
- Theil's U2: {format_metric(metrics_test, 'theil_u2')}
"""

    # Add LSTM-specific metrics if they exist
    if 'mase' in metrics_test or 'r2' in metrics_test:
        md += f"""
- MASE: {format_metric(metrics_test, 'mase')}
- R²: {format_metric(metrics_test, 'r2')}
- Adjusted R²: {format_metric(metrics_test, 'adj_r2')}
"""

    md += f"""
## Время обучения
- {train_time:.2f} секунд

## График прогноза

<img src=\"data:image/png;base64,{forecast_img_base64}\" alt=\"Прогноз\">

## График потерь

<img src=\"data:image/png;base64,{loss_img_base64}\" alt=\"Графики потерь\">

---
*Отчет сгенерирован автоматически системой TimeSeriesApp*
"""
    return md


def markdown_to_pdf(md_text: str) -> bytes:
    """
    Преобразует markdown в PDF (использует weasyprint или pdfkit).
    """
    # Преобразуем markdown в html
    if markdown2 is not None:
        html = markdown2.markdown(md_text)
    else:
        raise ImportError("markdown2 не установлен")
    # Преобразуем html в pdf
    if HTML is not None:
        pdf_bytes = HTML(string=html).write_pdf()
        return pdf_bytes
    elif pdfkit is not None:
        with tempfile.NamedTemporaryFile(suffix='.html', delete=False) as tmp_html:
            tmp_html.write(html.encode('utf-8'))
            tmp_html.flush()
            pdf_bytes = pdfkit.from_file(tmp_html.name, False)
        return pdf_bytes
    else:
        raise ImportError("Не установлен ни weasyprint, ни pdfkit")


def download_report_buttons(md_text: str, pdf_bytes: Optional[bytes] = None, md_filename: str = "report.md", pdf_filename: str = "report.pdf"):
    """
    Универсальная функция для отображения кнопок скачивания отчета в Streamlit.
    """
    st.download_button(
        label="Скачать отчет (.md)",
        data=md_text,
        file_name=md_filename,
        mime="text/markdown"
    )
    if pdf_bytes:
        st.download_button(
            label="Скачать отчет (.pdf)",
            data=pdf_bytes,
            file_name=pdf_filename,
            mime="application/pdf"
        )