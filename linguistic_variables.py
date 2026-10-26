"""
Лингвистические переменные для работы с нечеткими критериями.
"""

import re
from config import RAPIDFUZZ_AVAILABLE
from fuzzy_matching import fuzzy_match


class LinguisticVariable:
    """
    Класс для работы с лингвистическими переменными.
    Преобразует лингвистические термы (например, "не очень длинные") в числовые диапазоны.
    """
    
    def __init__(self, name, min_val, max_val):
        """
        :param name: Название переменной (например, "Длительность")
        :param min_val: Минимальное значение в данных
        :param max_val: Максимальное значение в данных
        """
        self.name = name
        self.min_val = min_val
        self.max_val = max_val
        self.range_val = max_val - min_val
        
        # Определяем термы и их числовые диапазоны
        # Используем процентили для более точного разбиения
        self.terms = self._define_terms()
    
    def _define_terms(self):
        """
        Определяет лингвистические термы и их числовые диапазоны.
        Теперь диапазоны строятся по данным респондентов (таблица μ_i(x_j)),
        а не по выбранным вручную процентилям.
        """
        # Точки x_j из таблицы респондентов (длительность спектакля)
        x_points = [60, 85, 100, 120, 130, 150, 170, 180, 195, 200, 215, 240]

        # Функции принадлежности μ_i(x_j) для каждого терма из таблицы
        membership_base = {
            'очень не длинные':             [1, 1, 0.25, 0, 0, 0, 0, 0, 0, 0, 0, 0],
            'не длинные':                   [0, 0, 0.75, 0.5, 0, 0, 0, 0, 0, 0, 0, 0],
            'не очень длинные':             [0, 0, 0, 0.5, 0.75, 0, 0, 0, 0, 0, 0, 0],
            'средней продолжительности':    [0, 0, 0, 0, 0.25, 1, 0.75, 0.25, 0.25, 0, 0, 0],
            'длинные':                      [0, 0, 0, 0, 0, 0, 0.25, 0.75, 0.5, 0.25, 0.25, 0],
            'очень длинные':                [0, 0, 0, 0, 0, 0, 0, 0, 0.25, 0.75, 0.75, 1],
        }

        # Добавляем синоним "недлинные" к тому же набору значений, что и "не длинные"
        membership_by_term = {
            'очень не длинные': membership_base['очень не длинные'],
            'очень недлинные': membership_base['очень не длинные'],
            'не длинные': membership_base['не длинные'],
            'недлинные': membership_base['не длинные'],
            'не очень длинные': membership_base['не очень длинные'],
            'средней продолжительности': membership_base['средней продолжительности'],
            'длинные': membership_base['длинные'],
            'очень длинные': membership_base['очень длинные'],
        }

        # Расчет границ через точки пересечения соседних функций μ (как на схеме)
        terms = self._calculate_intersection_ranges(x_points, membership_by_term)

        return terms

    def _calculate_intersection_ranges(self, x_points, membership_by_term):
        """
        Строит диапазоны через точки пересечения соседних термов (линейная интерполяция).
        Порядок термов фиксирован от "коротких" к "длинным".
        """
        base_order = [
            'очень не длинные',
            'не длинные',
            'не очень длинные',
            'средней продолжительности',
            'длинные',
            'очень длинные',
        ]

        # Проверяем длину массивов μ
        for term, mu_values in membership_by_term.items():
            if len(mu_values) != len(x_points):
                raise ValueError(
                    f'Длина массива μ для терма "{term}" не совпадает с количеством точек x_j'
                )

        # Находим точки пересечения соседних термов
        intersections = []
        for i in range(len(base_order) - 1):
            t_left = base_order[i]
            t_right = base_order[i + 1]
            mu_left = membership_by_term[t_left]
            mu_right = membership_by_term[t_right]
            x_star = self._find_intersection_point(x_points, mu_left, mu_right)
            if x_star is None:
                # Если пересечения нет, используем границу по опоре
                support_left = [x_points[idx] for idx, mu in enumerate(mu_left) if mu > 0]
                support_right = [x_points[idx] for idx, mu in enumerate(mu_right) if mu > 0]
                if not support_left or not support_right:
                    continue
                x_star = (support_left[-1] + support_right[0]) / 2
            intersections.append(x_star)

        # Собираем диапазоны для базовых термов
        terms = {}
        for idx, term in enumerate(base_order):
            mu_values = membership_by_term[term]
            support = [x_points[i] for i, mu in enumerate(mu_values) if mu > 0]
            if not support:
                continue

            if idx == 0:
                min_val = support[0]
                max_val = intersections[0] if intersections else support[-1]
            elif idx == len(base_order) - 1:
                min_val = intersections[-1] if intersections else support[0]
                max_val = support[-1]
            else:
                min_val = intersections[idx - 1]
                max_val = intersections[idx]

            terms[term] = {
                'min': min_val,
                'max': max_val,
                'operator': 'range',
                'value': (min_val, max_val),
            }

        # Синонимы: копируем диапазоны у базовых термов
        if 'не длинные' in terms:
            terms['недлинные'] = terms['не длинные']
        if 'очень не длинные' in terms:
            terms['очень недлинные'] = terms['очень не длинные']

        return terms

    def _find_intersection_point(self, x_points, mu_a, mu_b):
        """
        Ищет точку пересечения двух функций μ_A и μ_B линейной интерполяцией.
        Возвращает первую найденную x*, где μ_A(x)=μ_B(x).
        """
        for i in range(len(x_points) - 1):
            x1, x2 = x_points[i], x_points[i + 1]
            da = mu_a[i] - mu_b[i]
            db = mu_a[i + 1] - mu_b[i + 1]

            # Оба нуля — идём дальше
            if da == 0 and db == 0:
                continue
            # Точное равенство в узле
            if da == 0:
                return x1
            if db == 0:
                return x2

            # Пересечение по смене знака
            if da * db < 0:
                ratio = da / (da - db)
                return x1 + (x2 - x1) * ratio

        return None
    
    def get_term_range(self, term):
        """
        Получает числовой диапазон для лингвистического терма.
        :param term: Лингвистический терм (например, "не очень длинные")
        :return: Словарь с информацией о диапазоне или None, если терм не найден
        """
        # Нормализуем терм (приводим к нижнему регистру и убираем лишние пробелы)
        term_normalized = ' '.join(term.lower().split())
        
        # Прямой поиск
        if term_normalized in self.terms:
            return self.terms[term_normalized]
        
        # Нечеткий поиск с помощью fuzzy matching
        if RAPIDFUZZ_AVAILABLE:
            try:
                matched_term = fuzzy_match(term_normalized, list(self.terms.keys()), threshold=75)
                if matched_term:
                    return self.terms[matched_term]
            except Exception:
                pass
        
        return None
    
    def find_term_in_query(self, query):
        """
        Ищет лингвистический терм в запросе.
        :param query: Текст запроса
        :return: Найденный терм и его диапазон, или (None, None)
        """
        query_lower = query.lower()
        # Разбиваем запрос на слова для поиска целых слов/фраз
        words = query_lower.split()
        
        # ВАЖНО: Сначала проверяем многословные термы (более длинные), чтобы избежать
        # ситуации, когда "длинные" находится в "очень длинные"
        # Сортируем термы по длине (сначала самые длинные)
        sorted_terms = sorted(self.terms.items(), key=lambda x: len(x[0].split()), reverse=True)
        
        # Проверяем каждый терм (от самых длинных к самым коротким)
        for term, range_info in sorted_terms:
            term_words = term.split()
            # Если терм состоит из одного слова, ищем его как целое слово
            if len(term_words) == 1:
                # Ищем целое слово (не подстроку) - проверяем границы слов
                term_word = term_words[0]
                # Используем регулярное выражение для поиска целого слова
                pattern = r'\b' + re.escape(term_word) + r'\b'
                if re.search(pattern, query_lower):
                    return term, range_info
            else:
                # Если терм состоит из нескольких слов, ищем последовательность слов
                term_str = ' '.join(term_words)
                # Проверяем, содержится ли последовательность слов в запросе
                if term_str in query_lower:
                    # Дополнительная проверка: убеждаемся, что это не часть другого слова
                    # Ищем позицию терма в запросе
                    idx = query_lower.find(term_str)
                    if idx != -1:
                        # Проверяем, что перед и после терма стоят границы слов (пробелы или знаки препинания)
                        before_char = query_lower[idx - 1] if idx > 0 else ' '
                        after_idx = idx + len(term_str)
                        after_char = query_lower[after_idx] if after_idx < len(query_lower) else ' '
                        # Проверяем, что перед термом нет буквы (или это начало строки)
                        before_ok = idx == 0 or not before_char.isalpha()
                        # Проверяем, что после терма нет буквы (или это конец строки)
                        after_ok = after_idx >= len(query_lower) or not after_char.isalpha()
                        if before_ok and after_ok:
                            return term, range_info
        
        # Если не нашли точное совпадение, пробуем нечеткий поиск по целым фразам
        if RAPIDFUZZ_AVAILABLE:
            # Извлекаем потенциальные фразы из запроса (только целые последовательности слов)
            # Проверяем фразы разной длины (от 4 до 1 слова, от самых длинных к самым коротким)
            # Это важно, чтобы "очень длинные" находилось раньше, чем "длинные"
            for length in range(4, 0, -1):
                for i in range(len(words) - length + 1):
                    phrase = ' '.join(words[i:i+length])
                    # Используем отсортированный список термов (от самых длинных к самым коротким)
                    matched_term = fuzzy_match(phrase, [t[0] for t in sorted_terms], threshold=70)
                    if matched_term:
                        return matched_term, self.terms[matched_term]
        
        return None, None
    
    def get_filter_from_term(self, term):
        """
        Преобразует лингвистический терм в фильтр для pandas.
        :param term: Лингвистический терм
        :return: Словарь с информацией о фильтре или None
        """
        range_info = self.get_term_range(term)
        if not range_info:
            return None

        operator = range_info['operator']
        value = range_info['value']

        # На всякий случай упорядочиваем диапазон, если min/max пришли перепутанными
        if operator == 'range' and isinstance(value, (tuple, list)) and len(value) == 2:
            lo, hi = value
            if lo > hi:
                lo, hi = hi, lo
                range_info = {**range_info, 'value': (lo, hi), 'min': lo, 'max': hi}
                self.terms[' '.join(range_info.get('name', '').split())] = range_info
            return {
                'type': 'range',
                'min_filter': {'operator': '>=', 'value': lo},
                'max_filter': {'operator': '<=', 'value': hi}
            }
        else:
            return {
                'type': 'single',
                'operator': operator,
                'value': value
            }

