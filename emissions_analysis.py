import matplotlib.pyplot as plt
# load_boston() устарел и больше не используется
from sklearn.datasets import fetch_california_housing
import pandas as pd


def detect_emission(data, coef=1.5):
    """
    Функция для поиска выбросов по методу IQR.

    Входные данные:
        data (pandas.Series) -  Ряд чисел по которым определяются границы
        coef (float) - Коэффициент для IQR. Значение по классическому правилу равно 1.5

    Возвращает:
        (pandas.Series) - Ряд из True/False, где True - это выброс
    """
    # Определение границ
    Q1 = data.quantile(0.25)
    Q3 = data.quantile(0.75)
    IQR = Q3 - Q1
    lower = Q1 - coef * IQR
    upper = Q3 + coef * IQR

    # Числа вне границ являются выбросами
    return (data < lower) | (data > upper)


def create_graph(data, emissions, yname):
    """
    Функция для создания графика с выделением выбросов

    Входные данные :
        data (pandas.Series) - Ряд чисел, по которым строиться график.
        emissions (pandas.Series) - Ряд из True/False, где True — выброс.
        yname (str) - Название признака

    Возвращает:
        Ничего, только строит график
    """
    plt.figure(figsize=(16, 8))
    plt.scatter(range(len(data[emissions])), data[emissions], color="red", label="Выбросы")
    plt.scatter(range(len(data[~emissions])), data[~emissions])
    plt.xlabel("Географические зоны Калифорнии")
    plt.ylabel(yname)
    plt.title("Выбросы по IQR")
    plt.legend()
    plt.show()


def main():
    # Загружаем данные сразу в виде DataFrame
    data = fetch_california_housing(as_frame=True).frame

    # Ввод признака
    print("Доступные признаки:", ", ".join(data.columns))
    while True:
        name = input("Введите название признака ")
        if name in data.columns:
            break
        print("Неправильное название признака. Попробуйте еще раз.")

    # Берем данные по признаку
    med = data[name]

    # Проверка на пустые значения
    if med.isna().all():
        print("В выбранном признаке все значения отсутствуют")
        return

    # Проверка типа данных
    if not med.dtype.kind in "if":
        print("Для поиска выбросов данные должны быть числовыми")
        return

    # Находим выбросы по признаку
    # Выбросы обозначены как True, все остальное False
    emissions = detect_emission(med, 1.5)
    if emissions.sum() == 0:
        print("Выбросов не найдено")
        return
    
    # Вывод выбросов
    print(med[emissions])
    
    # Построение графика
    create_graph(med, emissions, name)


if __name__ == "__main__":
    main()