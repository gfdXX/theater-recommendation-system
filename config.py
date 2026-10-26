"""
Конфигурация и проверка зависимостей системы рекомендаций.
"""

import numpy as np
import pandas as pd
import re
import copy
from tabulate import tabulate
from collections import deque

# ==================================================================================
# ПРОВЕРКА И ИМПОРТ БИБЛИОТЕК
# ==================================================================================

# Импорт для нечеткого поиска (fuzzy matching)
try:
    from rapidfuzz import fuzz, process
    RAPIDFUZZ_AVAILABLE = True
except ImportError:
    try:
        from fuzzywuzzy import fuzz, process
        RAPIDFUZZ_AVAILABLE = True
    except ImportError:
        RAPIDFUZZ_AVAILABLE = False
        print("⚠️ rapidfuzz или fuzzywuzzy не установлены. Установите: pip install rapidfuzz")
        print("   (или: pip install fuzzywuzzy[speedup])")

# Импорты для морфологического и синтаксического анализа
PYMORPHY2_AVAILABLE = False
pymorphy2 = None
try:
    import pymorphy2
    # Пытаемся инициализировать, чтобы проверить совместимость
    try:
        _test_morph = pymorphy2.MorphAnalyzer()
        PYMORPHY2_AVAILABLE = True
    except (AttributeError, Exception) as e:
        # Ошибка совместимости (например, Python 3.11+ с старым pymorphy2)
        PYMORPHY2_AVAILABLE = False
        print(f"⚠️ pymorphy2 несовместим с текущей версией Python: {e}")
        print("💡 Для Google Colab выполните: !pip install --upgrade pymorphy2")
        print("   Система будет работать без морфологического анализа (с ограниченной функциональностью)")
except ImportError:
    PYMORPHY2_AVAILABLE = False
    print("⚠️ pymorphy2 не установлен. Для Google Colab: !pip install pymorphy2")
    print("   Система будет работать без морфологического анализа")

SPACY_AVAILABLE = False
nlp_spacy = None
try:
    import spacy
    SPACY_AVAILABLE = True
    # Пытаемся загрузить русскую модель
    try:
        nlp_spacy = spacy.load("ru_core_news_sm")
    except OSError:
        print("⚠️ Русская модель spaCy не найдена.")
        print("   Для Google Colab выполните: !python -m spacy download ru_core_news_sm")
        print("   Система будет работать без синтаксического анализа (с ограниченной функциональностью)")
        nlp_spacy = None
        SPACY_AVAILABLE = False
except ImportError:
    SPACY_AVAILABLE = False
    nlp_spacy = None
    print("⚠️ spacy не установлен. Для Google Colab: !pip install spacy")
    print("   Система будет работать без синтаксического анализа")

def print_library_status():
    """Выводит статус всех библиотек."""
    print("\n" + "="*70)
    print("📦 СТАТУС БИБЛИОТЕК:")
    print(f"   🔍 Нечеткий поиск (fuzzy matching): {'✅ Доступен' if RAPIDFUZZ_AVAILABLE else '❌ Не установлен (ОБЯЗАТЕЛЬНО!)'}")
    print(f"   📚 Морфологический анализ (pymorphy2): {'✅ Доступен' if PYMORPHY2_AVAILABLE else '⚠️ Недоступен (опционально)'}")
    print(f"   🔤 Синтаксический анализ (spacy): {'✅ Доступен' if SPACY_AVAILABLE else '⚠️ Недоступен (опционально)'}")
    print("="*70 + "\n")

