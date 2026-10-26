"""
Преобразование запросов на естественном языке в фильтры для pandas.
"""

import re
import numpy as np
import pandas as pd
from nlp_analyzers import MorphologicalAnalyzer, SyntacticAnalyzer
from fuzzy_matching import fuzzy_match, fuzzy_match_with_score
from config import RAPIDFUZZ_AVAILABLE


class QueryToFilterConverter:
    """Преобразует сложный текстовый запрос в логическое выражение (фильтр) для Pandas."""

    def __init__(self, df_columns, duration_linguistic=None):
        self.df_columns = df_columns
        self.morph_analyzer = MorphologicalAnalyzer()
        self.syntax_analyzer = SyntacticAnalyzer()
        self.duration_linguistic = duration_linguistic  # Лингвистическая переменная для длительности

    COLUMN_MAP = {
        'жанр': 'Жанр', 'театр': 'Театр', 'стиль': 'Стиль', 'язык': 'Язык исполнения',
        'аудитория': 'Целевая аудитория', 'рейтинг': 'Рейтинг', 'длительность': 'Длительность (мин)',
        'режиссер': 'Режиссер', 'антрактов': 'Количество антрактов',
        'платные': 'Платный?', 'семейные': 'Семейный?', 'детские': 'Для детей?',
        'год': 'Год постановки', 'награды': 'Награды', 'теги': 'Теги', 'темы': 'Темы',
        'открытом воздухе': 'Проводится на открытом воздухе?', 'открытый воздух': 'Проводится на открытом воздухе?'
    }
    OPERATOR_MAP = {
        'выше': '>', 'больше': '>', 'менее': '<', 'ниже': '<', 'до': '<=', 'после': '>=',
        'равен': '==', 'идут в': '==', 'в': '==', 'являются': '=='
    }

    def find_column_name(self, word):
        """Находит имя колонки по части слова."""
        word = word.lower()
        for key, col in self.COLUMN_MAP.items():
            if key in word:
                return col
        return None

    def _convert_to_python_type(self, val):
        """Преобразует numpy/pandas типы в Python типы."""
        if isinstance(val, (np.integer, np.floating)):
            return val.item()
        elif isinstance(val, np.ndarray):
            if val.size == 1:
                return val.item()
            elif val.size > 0:
                return val[0]
            else:
                return None
        elif isinstance(val, pd.Series):
            if len(val) > 0:
                return val.iloc[0]
            else:
                return None
        else:
            return val
    
    def find_value_in_query(self, query, column_name, unique_values):
        """Ищет конкретное значение критерия в запросе."""
        query_lower = query.lower()

        # Нормализуем запрос для лучшего поиска
        normalized_query = self.morph_analyzer.normalize_text(query)
        normalized_query_lower = normalized_query.lower()

        # Используем синтаксический анализ для поиска именных групп
        noun_phrases = self.syntax_analyzer.find_noun_phrases(query)

        if column_name in ['Рейтинг', 'Длительность (мин)', 'Количество антрактов']:
            # Ищем числовое значение
            if column_name == 'Рейтинг':
                patterns = [
                    r'рейтинг\s+(?:выше|больше|>\s*)(\d+[\.,]?\d*)',
                    r'рейтинг\s+(?:ниже|меньше|<\s*)(\d+[\.,]?\d*)',
                    r'с\s+рейтингом\s+(\d+[\.,]?\d*)',
                    r'рейтинг\s+(\d+[\.,]?\d*)',
                ]
                for pattern in patterns:
                    match = re.search(pattern, query_lower)
                    if match:
                        try:
                            return float(match.group(1).replace(',', '.'))
                        except ValueError:
                            continue
                match = re.search(r'(\d+[\.,]?\d*)', query_lower)
                if match:
                    try:
                        return float(match.group(1).replace(',', '.'))
                    except ValueError:
                        return None
            else:
                match = re.search(r'(\d+[\.,]?\d*)', query_lower)
                if match:
                    try:
                        return float(match.group(1).replace(',', '.'))
                    except ValueError:
                        return None

        # Ищем строковое значение (например, "Опера", название театра)
        if column_name == 'Театр':
            # Сначала пробуем точное совпадение
            for val in unique_values:
                val_lower = str(val).lower()
                normalized_val = self.morph_analyzer.normalize_text(str(val))
                val_words = set(normalized_val.split())
                query_words = set(normalized_query_lower.split())

                # Убираем стоп-слова
                stop_words = {'в', 'и', 'на', 'с', 'театр', 'театре', 'театра', 'им', 'имени', 'найди', 'постановки', 'спектакли', 'покажи', 'выведи'}
                val_words = {w for w in val_words if w not in stop_words and len(w) > 2}
                query_words = {w for w in query_words if w not in stop_words and len(w) > 2}

                # Если название театра полностью содержится в запросе
                if normalized_val in normalized_query_lower or val_lower in query_lower:
                    return self._convert_to_python_type(val)

                # Проверяем точное совпадение уникальных слов театра
                unique_theatre_words = [w for w in val_words if w not in ['музыкальный', 'театр', 'театра', 'имени', 'им']]
                if unique_theatre_words:
                    for unique_word in unique_theatre_words:
                        pattern = r'\b' + re.escape(unique_word) + r'\b'
                        if re.search(pattern, normalized_query_lower):
                            return self._convert_to_python_type(val)

                # Если есть значительное пересечение ключевых слов
                intersection = val_words.intersection(query_words)
                if len(intersection) >= 2 or (len(val_words) <= 3 and len(intersection) >= 1):
                    if any(w in normalized_query_lower for w in unique_theatre_words):
                        return self._convert_to_python_type(val)

                # Используем синтаксический анализ
                if noun_phrases:
                    try:
                        for phrase in noun_phrases:
                            phrase_lemma = phrase.get('root_lemma', '')
                            if phrase_lemma and (phrase_lemma in normalized_val or phrase_lemma in val_lower):
                                return self._convert_to_python_type(val)
                    except Exception:
                        pass
            
            # Если точное совпадение не найдено, используем fuzzy matching для театров
            if RAPIDFUZZ_AVAILABLE and len(unique_values) > 0:
                values_dict = {str(val): val for val in unique_values}
                stop_words = {'в', 'и', 'на', 'с', 'театр', 'театре', 'театра', 'им', 'имени', 'найди', 'постановки', 'спектакли', 'найди', 'покажи', 'выведи'}
                query_keywords = [w for w in normalized_query_lower.split() if w not in stop_words and len(w) > 2]
                
                if query_keywords:
                    best_match = None
                    best_score = 0
                    
                    for keyword in query_keywords:
                        if len(keyword) > 3:
                            matched_value, score = fuzzy_match_with_score(keyword, values_dict, threshold=95)
                            if matched_value and score > best_score:
                                best_match = matched_value
                                best_score = score
                    
                    if best_match:
                        return self._convert_to_python_type(values_dict[best_match])
                    
                    query_keywords_str = ' '.join(query_keywords)
                    matched_value = fuzzy_match(query_keywords_str, values_dict, threshold=90)
                    if matched_value:
                        return self._convert_to_python_type(values_dict[matched_value])
                
                matched_value = fuzzy_match(normalized_query_lower, values_dict, threshold=90)
                if matched_value:
                    return self._convert_to_python_type(values_dict[matched_value])

        # Для других колонок используем поиск с морфологическим анализом и fuzzy matching
        for val in unique_values:
            val_lower = str(val).lower()
            normalized_val = self.morph_analyzer.normalize_text(str(val))

            if val_lower in query_lower or normalized_val in normalized_query_lower:
                return self._convert_to_python_type(val)

            if len(val_lower) > 5:
                if val_lower[:5] in query_lower or normalized_val[:5] in normalized_query_lower:
                    words = normalized_val.split()
                    if len(words) > 1 and all(w in normalized_query_lower for w in words[:2]):
                        return self._convert_to_python_type(val)

            if noun_phrases:
                try:
                    for phrase in noun_phrases:
                        phrase_lemma = phrase.get('root_lemma', '')
                        if phrase_lemma and (phrase_lemma in normalized_val or phrase_lemma in val_lower):
                            return self._convert_to_python_type(val)
                except Exception:
                    pass
        
        # Если точное совпадение не найдено, используем fuzzy matching
        if RAPIDFUZZ_AVAILABLE and column_name not in ['Рейтинг', 'Длительность (мин)', 'Количество антрактов', 'Платный?', 'Семейный?', 'Для детей?']:
            if len(unique_values) > 0:
                values_dict = {str(val): val for val in unique_values}
                query_keywords = [w for w in normalized_query_lower.split() if len(w) > 2]
                
                for keyword in query_keywords:
                    if len(keyword) > 3:
                        threshold = 75
                        matched_value = fuzzy_match(keyword, values_dict, threshold=threshold)
                        if matched_value:
                            return self._convert_to_python_type(values_dict[matched_value])
                
                matched_value = fuzzy_match(normalized_query_lower, values_dict, threshold=80)
                if matched_value:
                    return self._convert_to_python_type(values_dict[matched_value])

        # Обработка бинарных колонок
        if column_name in ['Платный?', 'Семейный?', 'Для детей?']:
            if 'не платные' in query_lower or 'не являются платными' in query_lower:
                return 'Нет'
            elif 'платные' in query_lower or 'являются платными' in query_lower:
                return 'Да'

        # Если точный поиск не дал результатов, используем fuzzy matching для строковых значений
        if column_name not in ['Рейтинг', 'Длительность (мин)', 'Количество антрактов', 'Платный?', 'Семейный?', 'Для детей?']:
            unique_values_valid = False
            if unique_values is not None:
                try:
                    unique_values_valid = len(unique_values) > 0
                except (TypeError, AttributeError):
                    unique_values_valid = True
            
            if RAPIDFUZZ_AVAILABLE and unique_values_valid:
                values_dict = {str(val): val for val in unique_values}
                stop_words = {'в', 'и', 'на', 'с', 'театр', 'театре', 'театра', 'им', 'имени', 'жанр', 'стиль', 'язык', 'аудитория', 'режиссер'}
                query_keywords = [w for w in normalized_query_lower.split() if w not in stop_words and len(w) > 2]
                
                if query_keywords:
                    query_keywords_str = ' '.join(query_keywords)
                    threshold = 70 if len(query_keywords_str) > 5 else 65
                    matched_value = fuzzy_match(query_keywords_str, values_dict, threshold=threshold)
                    if matched_value:
                        return self._convert_to_python_type(values_dict[matched_value])
                
                matched_value = fuzzy_match(normalized_query, values_dict, threshold=70)
                if matched_value:
                    return self._convert_to_python_type(values_dict[matched_value])

        return None

    def get_query_parts(self, query, df):
        """Парсит запрос и составляет список фильтров."""
        filters = []
        query_lower = query.lower()
        query_original = query
        
        # Проверяем наличие лингвистических термов для длительности
        has_duration_context = any(word in query_lower for word in ['длительность', 'длительностью', 'продолжительность', 'продолжительностью', 'минут', 'минуты'])
        has_duration_terms = False
        duration_term_patterns = [
            r'\bдлинные\b', r'\bнедлинные\b', r'\bне длинные\b', r'\bне очень длинные\b',
            r'\bочень длинные\b', r'\bочень не длинные\b', r'\bочень недлинные\b', r'\bсредней продолжительности\b'
        ]
        for pattern in duration_term_patterns:
            if re.search(pattern, query_lower):
                has_duration_terms = True
                break
        
        if self.duration_linguistic and (has_duration_context or has_duration_terms):
            term, range_info = self.duration_linguistic.find_term_in_query(query)
            if term and range_info:
                filter_info = self.duration_linguistic.get_filter_from_term(term)
                if filter_info:
                    if filter_info['type'] == 'range':
                        filters.append({
                            'column': 'Длительность (мин)',
                            'operator': filter_info['min_filter']['operator'],
                            'value': filter_info['min_filter']['value'],
                            'logic': 'AND'
                        })
                        filters.append({
                            'column': 'Длительность (мин)',
                            'operator': filter_info['max_filter']['operator'],
                            'value': filter_info['max_filter']['value'],
                            'logic': 'AND'
                        })
                    else:
                        filters.append({
                            'column': 'Длительность (мин)',
                            'operator': filter_info['operator'],
                            'value': filter_info['value'],
                            'logic': 'AND'
                        })

        is_theatre_in_query = any(phrase in query_lower for phrase in [
            'идут в', 'в театре', 'театр им.', 'музыкальный театр',
            'театр им', 'театр пушкина', 'театр моссовета', 'мхт',
            'театр эстрады', 'новая опера', 'современник', 'станиславского',
            'немировича', 'данченко', 'чехова', 'моссовета'
        ])

        query_clean = query_lower.replace(';', ',').replace(':', ',').replace('?', '')
        has_theatre_context = any(phrase in query_clean for phrase in ['в театре', 'театр', 'театре'])
        words_count = len(query_clean.split())
        
        if has_theatre_context and words_count > 6:
            parts = [query_clean]
        else:
            parts = re.split(r'\s+(и|или|а)\s+', query_clean)

        if len(parts) == 1:
            parts = [query_clean]
            process_full_query = True
        else:
            process_full_query = False

        duration_linguistic_found = any(f.get('column') == 'Длительность (мин)' for f in filters)

        for part in parts:
            part = part.strip()
            if part in ['и', 'или', 'а', '']: continue

            col_match = None
            is_theatre_column = any(phrase in part for phrase in [
                'идут в', 'в театре', 'театр им.', 'музыкальный театр',
                'театр им', 'театр пушкина', 'театр моссовета', 'мхт',
                'театр эстрады', 'новая опера', 'современник', 'станиславского',
                'немировича', 'данченко', 'чехова', 'моссовета'
            ]) or is_theatre_in_query

            genre_values = ['опера', 'оперы', 'мюзикл', 'мюзиклы', 'балет', 'комедия']
            if not is_theatre_column:
                genre_values.append('театр')

            for genre_val in genre_values:
                if genre_val in part:
                    col_match = 'Жанр'
                    if genre_val in ['опера', 'оперы']:
                        value = 'Опера'
                    elif genre_val in ['мюзикл', 'мюзиклы']:
                        value = 'Мьюзикл'
                    elif genre_val == 'балет':
                        value = 'Балет'
                    elif genre_val == 'театр':
                        value = 'Театр'
                    elif genre_val == 'комедия':
                        value = 'Комедия'
                    else:
                        value = None

                    if value is not None:
                        value = self._convert_to_python_type(value)
                        filters.append({
                            'column': col_match,
                            'operator': '==',
                            'value': value,
                            'logic': 'AND'
                        })
                    break

            if not col_match:
                if 'стиль' in part or 'в стиле' in part:
                    col_match = 'Стиль'
                elif is_theatre_column or ('театр' in part and any(phrase in part for phrase in ['идут', 'в театре', 'театр им', 'музыкальный'])):
                    col_match = 'Театр'
                elif 'рейтинг' in part:
                    col_match = 'Рейтинг'
                elif 'длительность' in part and not duration_linguistic_found:
                    col_match = 'Длительность (мин)'
                elif 'антракт' in part:
                    col_match = 'Количество антрактов'
                elif 'год' in part:
                    col_match = 'Год постановки'
                else:
                    for col_word, col_name in self.COLUMN_MAP.items():
                        if col_word in part:
                            col_match = col_name
                            break

                if not col_match and not is_theatre_column:
                    genre_values_check = ['опера', 'оперы', 'мюзикл', 'мюзиклы', 'балет', 'комедия']
                    if not is_theatre_column:
                        genre_values_check.append('театр')

                    for genre_val in genre_values_check:
                        if genre_val in part:
                            col_match = 'Жанр'
                            if genre_val in ['опера', 'оперы']:
                                value = 'Опера'
                            elif genre_val in ['мюзикл', 'мюзиклы']:
                                value = 'Мьюзикл'
                            elif genre_val == 'балет':
                                value = 'Балет'
                            elif genre_val == 'театр':
                                value = 'Театр'
                            elif genre_val == 'комедия':
                                value = 'Комедия'
                            else:
                                value = None

                            if value is not None:
                                value = self._convert_to_python_type(value)
                                filters.append({
                                    'column': col_match,
                                    'operator': '==',
                                    'value': value,
                                    'logic': 'AND'
                                })
                            break

            if not col_match:
                continue

            op = '=='
            for op_word, op_symbol in self.OPERATOR_MAP.items():
                if op_word in part:
                    op = op_symbol
                    break

            unique_values = df[col_match].dropna().unique()
            
            if col_match == 'Театр':
                part_clean = part
                if 'в театре' in part:
                    idx = part.find('в театре')
                    part_clean = part[idx + len('в театре'):].strip()
                elif 'театре' in part:
                    idx = part.find('театре')
                    part_clean = part[idx + len('театре'):].strip()
                elif 'театр' in part:
                    idx = part.find('театр')
                    part_clean = part[idx + len('театр'):].strip()
                
                if part_clean:
                    value = self.find_value_in_query(part_clean, col_match, unique_values)
                
                if value is None:
                    value = self.find_value_in_query(query_original, col_match, unique_values)
            elif col_match == 'Стиль':
                part_clean = part
                if 'в стиле' in part:
                    idx = part.find('в стиле')
                    part_clean = part[idx + len('в стиле'):].strip()
                elif 'стиле' in part:
                    idx = part.find('стиле')
                    part_clean = part[idx + len('стиле'):].strip()
                elif 'стиль' in part:
                    idx = part.find('стиль')
                    part_clean = part[idx + len('стиль'):].strip()
                if part_clean:
                    value = self.find_value_in_query(part_clean, col_match, unique_values)
                if value is None:
                    value = self.find_value_in_query(query_original, col_match, unique_values)
            else:
                value = self.find_value_in_query(part, col_match, unique_values)
            
            if value is None and col_match == 'Стиль':
                part_clean = part
                if 'в стиле' in part:
                    idx = part.find('в стиле')
                    part_clean = part[idx + len('в стиле'):].strip()
                elif 'стиле' in part:
                    idx = part.find('стиле')
                    part_clean = part[idx + len('стиле'):].strip()
                elif 'стиль' in part:
                    idx = part.find('стиль')
                    part_clean = part[idx + len('стиль'):].strip()
                if part_clean:
                    value = self.find_value_in_query(part_clean, col_match, unique_values)
                if value is None:
                    value = self.find_value_in_query(query_original, col_match, unique_values)
            elif value is None and col_match == 'Театр':
                part_clean = part
                if 'в театре' in part:
                    idx = part.find('в театре')
                    part_clean = part[idx + len('в театре'):].strip()
                elif 'театре' in part:
                    idx = part.find('театре')
                    part_clean = part[idx + len('театре'):].strip()
                elif 'театр' in part:
                    idx = part.find('театр')
                    part_clean = part[idx + len('театр'):].strip()
                
                if part_clean:
                    value = self.find_value_in_query(part_clean, col_match, unique_values)
                
                if value is None:
                    value = self.find_value_in_query(query_original, col_match, unique_values)
            elif value is None and col_match == 'Режиссер':
                value = self.find_value_in_query(query_original, col_match, unique_values)
            elif value is None and col_match == 'Рейтинг':
                value = self.find_value_in_query(query_original, col_match, unique_values)
            
            col_match_valid = bool(col_match) if col_match is not None else False
            value_valid = value is not None
            
            if col_match_valid and value_valid:
                if isinstance(value, (np.integer, np.floating)):
                    value = value.item()
                elif isinstance(value, np.ndarray):
                    if value.size == 1:
                        value = value.item()
                    elif value.size > 0:
                        value = value[0]
                    else:
                        continue
                elif isinstance(value, pd.Series):
                    if len(value) > 0:
                        value = value.iloc[0]
                    else:
                        continue
                
                value_str = str(value) if not isinstance(value, (int, float)) else value
                
                try:
                    filter_exists = any(
                        f.get('column') == col_match and f.get('operator') == op and
                        (str(f.get('value')) == value_str or 
                         (col_match == 'Театр' and str(f.get('value')).lower() in str(value_str).lower()))
                        for f in filters
                    )
                    if not filter_exists:
                        filters.append({
                            'column': col_match,
                            'operator': op,
                            'value': value,
                            'logic': 'AND'
                        })
                except Exception:
                    filters.append({
                        'column': col_match,
                        'operator': op,
                        'value': value,
                        'logic': 'AND'
                    })

        return filters

