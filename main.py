"""
Главный файл для запуска системы рекомендаций.
Устанавливает необходимые библиотеки и запускает интерактивную систему.
"""

import subprocess
import sys
import os

# ==================================================================================
# УСТАНОВКА БИБЛИОТЕК
# ==================================================================================

def install_package(package):
    """Устанавливает пакет через pip."""
    try:
        subprocess.check_call([sys.executable, "-m", "pip", "install", package])
        return True
    except subprocess.CalledProcessError:
        return False

def install_requirements():
    """Устанавливает все необходимые библиотеки."""
    print("📦 Установка необходимых библиотек...")
    print("=" * 70)
    
    packages = [
        ("rapidfuzz", "rapidfuzz"),  # ОБЯЗАТЕЛЬНО для работы fuzzy matching
        ("pymorphy2", "pymorphy2"),  # Опционально
        ("spacy", "spacy"),  # Опционально
    ]
    
    for package_name, pip_name in packages:
        print(f"Установка {package_name}...")
        if install_package(pip_name):
            print(f"✅ {package_name} установлен")
        else:
            print(f"⚠️ Не удалось установить {package_name}")
    
    # Установка русской модели spaCy
    print("\nЗагрузка русской модели spaCy...")
    try:
        subprocess.check_call([sys.executable, "-m", "spacy", "download", "ru_core_news_sm"])
        print("✅ Русская модель spaCy загружена")
    except subprocess.CalledProcessError:
        print("⚠️ Не удалось загрузить русскую модель spaCy")
        print("   Система будет работать без синтаксического анализа")
    
    print("=" * 70 + "\n")

# ==================================================================================
# ЗАГРУЗКА ДАННЫХ И ИНИЦИАЛИЗАЦИЯ
# ==================================================================================

import pandas as pd
import numpy as np
from tabulate import tabulate
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics.pairwise import euclidean_distances, manhattan_distances, cosine_distances
from sklearn.preprocessing import StandardScaler, MinMaxScaler
import matplotlib.pyplot as plt
import seaborn as sns

# Импортируем модули системы
from config import print_library_status
from natural_language_interface import NaturalLanguageInterface

def load_data():
    """Загружает данные из CSV файла."""
    try:
        df = pd.read_csv('test.csv', decimal=',')
        print("📊 ДАННЫЕ ЗАГРУЖЕНЫ")
        print("=" * 30)
        print(f"Количество спектаклей: {len(df)}")
        print(f"Колонки: {list(df.columns)}")
        print("\n🎭 ПЕРВЫЕ 5 СПЕКТАКЛЕЙ:")
        print(tabulate(df.head()[['ID', 'Название постановки', 'Жанр', 'Рейтинг', 'Театр']],
                      headers='keys', tablefmt='grid'))
        return df
    except FileNotFoundError:
        print("❌ Ошибка: файл test.csv не найден!")
        print("   Убедитесь, что файл test.csv находится в текущей директории.")
        sys.exit(1)
    except Exception as e:
        print(f"❌ Ошибка при загрузке данных: {e}")
        sys.exit(1)

def main():
    """Главная функция для запуска системы."""
    print("\n" + "=" * 70)
    print("🚀 СИСТЕМА РЕКОМЕНДАЦИЙ ДЛЯ ТЕАТРАЛЬНЫХ ПОСТАНОВОК")
    print("=" * 70 + "\n")
    
    # Проверяем, нужно ли устанавливать библиотеки
    install_required = input("Установить необходимые библиотеки? (y/n): ").strip().lower()
    if install_required == 'y':
        install_requirements()
    
    # Выводим статус библиотек
    print_library_status()
    
    # Загружаем данные
    df = load_data()
    
    print("\n✅ Все библиотеки импортированы!")
    print("🚀 Система готова к работе!")
    
    # Создаем и запускаем интерфейс
    interface = NaturalLanguageInterface(df)
    interface.run()

if __name__ == "__main__":
    main()

