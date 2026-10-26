"""
Функции для нечеткого поиска (fuzzy matching) с использованием расстояния Левенштейна.
"""

from config import RAPIDFUZZ_AVAILABLE

if RAPIDFUZZ_AVAILABLE:
    try:
        from rapidfuzz import fuzz, process
    except ImportError:
        from fuzzywuzzy import fuzz, process


def fuzzy_match(query, choices, threshold=75, scorer=None):
    """
    Находит наиболее похожее значение из списка choices для запроса query.
    Использует расстояние Левенштейна для обработки опечаток.
    
    :param query: Строка запроса (может содержать опечатки)
    :param choices: Список или словарь возможных значений
    :param threshold: Минимальный порог схожести (0-100)
    :param scorer: Функция для вычисления схожести (по умолчанию fuzz.ratio)
    :return: Наиболее похожее значение или None, если ничего не найдено
    """
    if not RAPIDFUZZ_AVAILABLE:
        return None
    
    if scorer is None:
        scorer = fuzz.ratio
    
    if isinstance(choices, dict):
        choices_list = list(choices.keys())
    else:
        choices_list = list(choices)
    
    if not choices_list:
        return None
    
    try:
        result = process.extractOne(str(query), choices_list, scorer=scorer, score_cutoff=threshold)
        if result:
            return result[0]
    except Exception:
        pass
    
    return None


def fuzzy_match_with_score(query, choices, threshold=75, scorer=None):
    """
    Находит наиболее похожее значение из списка choices для запроса query и возвращает score.
    
    :param query: Строка запроса (может содержать опечатки)
    :param choices: Список или словарь возможных значений
    :param threshold: Минимальный порог схожести (0-100)
    :param scorer: Функция для вычисления схожести (по умолчанию fuzz.ratio)
    :return: Кортеж (наиболее похожее значение, score) или (None, 0)
    """
    if not RAPIDFUZZ_AVAILABLE:
        return (None, 0)
    
    if scorer is None:
        scorer = fuzz.ratio
    
    if isinstance(choices, dict):
        choices_list = list(choices.keys())
    else:
        choices_list = list(choices)
    
    if not choices_list:
        return (None, 0)
    
    try:
        result = process.extractOne(str(query), choices_list, scorer=scorer, score_cutoff=threshold)
        if result:
            return (result[0], result[1])
    except Exception:
        pass
    
    return (None, 0)

