import tkinter as tk
import moex_api
import matplotlib.pyplot as plt
from matplotlib.figure import Figure
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg
import pandas as pd
import pywt
import numpy as np
from mpl_interactions import zoom_factory, panhandler

app = tk.Tk()
app.title("Прогнозирование цен акций")

selected_stock = "LKOH"
selected_wavelet = "haar"
selected_interval = 7
start_date = "2023-01-01"
end_date = "2023-10-01"
levels = 5

fig, (ax1, ax2) = plt.subplots(nrows=2, sharex=True, figsize=(8, 8))

canvas = FigureCanvasTkAgg(fig, master=app)
canvas_widget = canvas.get_tk_widget()
canvas_widget.pack(side=tk.LEFT, fill=tk.BOTH, expand=True)  
previous_lines = []

def update_plot(data):
    print(data)
    ax1.clear()
    ax2.clear()

    waveletData = pywt.wavedec(data['close'], wavelet=selected_wavelet, level=levels)

    for i in range(1, levels + 1):
        line = ax2.plot(pywt.upcoef('a', waveletData[i], selected_wavelet, level=i, take=len(data['close'])), label=f'Уровень {i}')
        previous_lines.append(line)


    ax1.plot(data['begin'], data['close'], label="Цена акции")
    ax1.set_xlabel("Время")
    ax1.set_ylabel("Цена акции")
    ax1.set_title(f"График цены акции {selected_stock}")

    ax2.set_title(f"Графики вейвлетов")

    x_ticks_indices = np.linspace(0, len(data['begin']) - 1, 10, dtype=int)
    x_ticks_labels = [pd.to_datetime(data['begin'].iloc[i]).strftime("%m-%d") for i in x_ticks_indices]
    ax1.set_xticks(x_ticks_indices)
    ax1.set_xticklabels(x_ticks_labels)
    ax2.set_xticks(x_ticks_indices)
    ax2.set_xticklabels(x_ticks_labels)

    ax2.set_xlim(0, len(data['begin']))

    ax2.legend()

    zoom_factory(ax1, base_scale=1.1)
    zoom_factory(ax2, base_scale=1.1)
    panhandler(fig)

    canvas.draw()
    

# Функция для обработки события выбора акции
def select_stock():
    global selected_stock
    selected_stock = stock_var.get()
    data = moex_api.get_data(tiker=selected_stock, interval=selected_interval)
    update_plot(data)
    print(f"Выбрана акция: {selected_stock}")

# Функция для обработки события выбора типа вейвлет-анализа
def select_wavelet():
    global selected_wavelet
    selected_wavelet = wavelet_var.get()
    update_plot(data)
    print(f"Выбран тип вейвлет-анализа: {selected_wavelet}")

# Функция для обработки события выбора временного интервала
def select_time_interval():
    global selected_interval
    selected_interval = intervals_mapping[interval_var.get()]
    data = moex_api.get_data(tiker=selected_stock, interval=selected_interval)
    update_plot(data)
    print(f"Выбран временной интервал: {selected_interval} сек")

# Функция для обработки события выбора диапазона дат
def select_dates():
    global start_date, end_date
    start_date = start_date_entry.get()
    end_date = end_date_entry.get()
    data = moex_api.get_data(tiker=selected_stock, interval=selected_interval, start=start_date, end=end_date)
    update_plot(data)
    print(f"Выбран диапазон дат: {start_date} - {end_date}")

# Функция для обработки события выбора уровня
def select_levels():
    global levels
    levels = int(levels_entry.get())
    update_plot(data)
    print(f"Выбран уровень: {levels}")

# Список выбора акций
stock_label = tk.Label(app, text="Выберите акцию:")
stock_label.pack()
stocks = ["LKOH", "SBER", "GAZP", "UNAC", "MOEX"] 
stock_var = tk.StringVar()
stock_var.set(stocks[0])
stock_menu = tk.OptionMenu(app, stock_var, *stocks)
stock_menu.pack()

# Кнопка для выбора акции
select_stock_button = tk.Button(app, text="Выбрать акцию", command=select_stock)
select_stock_button.pack()

# Список выбора типа вейвлет-анализа
wavelet_label = tk.Label(app, text="Выберите тип вейвлет-анализа:")
wavelet_label.pack()
wavelets = ["haar", "bior3.3", "coif3", "sym15"]  
wavelet_var = tk.StringVar()
wavelet_var.set(wavelets[0])
wavelet_menu = tk.OptionMenu(app, wavelet_var, *wavelets)
wavelet_menu.pack()

# Кнопка для выбора типа вейвлет-анализа
select_wavelet_button = tk.Button(app, text="Выбрать вейвлет", command=select_wavelet)
select_wavelet_button.pack()

# Список выбора временного интервала
interval_label = tk.Label(app, text="Выберите временной интервал:")
interval_label.pack()
intervals_mapping = {
    "1 сек": 1,
    "10 мин": 10,
    "1 час": 60,
    "1 день": 24,
    "1 неделя": 7,
    "1 месяц": 31,
    "1 квартал": 4,
}
interval_var = tk.StringVar()
interval_var.set("1 неделя")
interval_menu = tk.OptionMenu(app, interval_var, *intervals_mapping.keys())
interval_menu.pack()

# Кнопка для выбора временного интервала
select_interval_button = tk.Button(app, text="Выбрать интервал", command=select_time_interval)
select_interval_button.pack()

# Поля ввода для даты начала и конца интервала
start_date_label = tk.Label(app, text="Дата начала:")
start_date_label.pack()
start_date_entry = tk.Entry(app)
start_date_entry.insert(0, start_date)
start_date_entry.pack()

end_date_label = tk.Label(app, text="Дата конца:")
end_date_label.pack()
end_date_entry = tk.Entry(app)
end_date_entry.insert(0, end_date)
end_date_entry.pack()

# Кнопка для выбора даты
select_dates_levels_button = tk.Button(app, text="Применить", command=select_dates)
select_dates_levels_button.pack()

# Поле ввода для уровней вейвлет-преобразований
levels_label = tk.Label(app, text="Уровни вейвлет-преобразований:")
levels_label.pack()
levels_entry = tk.Entry(app)
levels_entry.insert(0, levels)
levels_entry.pack()

# Кнопка для выбора уровней вейвлет-анализа
select_dates_levels_button = tk.Button(app, text="Применить", command=select_levels)
select_dates_levels_button.pack()

# Запуск главного цикла приложения
if __name__ == "__main__":
    data = moex_api.get_data(tiker=selected_stock, interval=selected_interval, start=start_date, end=end_date)
    update_plot(data)
    app.mainloop()
