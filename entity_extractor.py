"""
Извлечение сущностей (ID, критерии) из запросов на естественном языке.
"""

import re
import pandas as pd
from nlp_analyzers import MorphologicalAnalyzer, SyntacticAnalyzer
from fuzzy_matching import fuzzy_match, fuzzy_match_with_score
from config import RAPIDFUZZ_AVAILABLE


class EntityExtractor:
    """Извлекает числовые ID или критерии фильтрации."""

    def __init__(self):
        self.current_df = None  # Инициализируем атрибут
        self.morph_analyzer = MorphologicalAnalyzer()
        self.syntax_analyzer = SyntacticAnalyzer()

    def extract_id(self, tokenized_text, raw_text):
        """Извлекает числовой ID."""
        match = re.search(r'(id|№|номер|спектакль|постановк\w*)\s*(\d+)', raw_text, re.I)
        if match:
            return int(match.group(2))

        match = re.search(r'\s(\d+)\s*$', raw_text)
        if match:
            return int(match.group(1))

        return None

    # НОВЫЙ МЕТОД: Поиск ID по названию постановки (с поддержкой fuzzy matching)
    def find_id_by_title(self, query, df):
        """Ищет ID по части названия постановки в запросе с поддержкой опечаток."""
        query_parts = query.split()

        # Нормализуем запрос с помощью морфологического анализа
        normalized_query = self.morph_analyzer.normalize_text(query)
        normalized_query_parts = normalized_query.split()

        # Используем синтаксический анализ для поиска именных групп
        noun_phrases = self.syntax_analyzer.find_noun_phrases(query)
        query_entities = [phrase.get('root_lemma', '') for phrase in noun_phrases if phrase.get('root_lemma')]

        # Собираем все названия для fuzzy matching
        titles_dict = {}
        for index, row in df.iterrows():
            title = str(row['Название постановки'])
            titles_dict[title] = row['ID']

        # Сначала пробуем точный поиск
        for index, row in df.iterrows():
            title = str(row['Название постановки']).lower()
            normalized_title = self.morph_analyzer.normalize_text(title)
            normalized_title_parts = normalized_title.split()

            # Проверка, содержится ли название целиком или ключевые слова из запроса
            if title in query.lower() or normalized_title in normalized_query:
                return row['ID']

            # Проверка на частичное совпадение с нормализованными формами
            title_words_set = set(normalized_title_parts)
            query_set = set(normalized_query_parts)

            # Если есть пересечение ключевых слов, и это не стоп-слова
            intersection = title_words_set.intersection(query_set)
            if len(intersection) > 1:
                return row['ID']

            # Проверка по именным группам из синтаксического анализа
            if query_entities:
                try:
                    title_entities = self.syntax_analyzer.find_noun_phrases(title)
                    title_entity_lemmas = [phrase.get('root_lemma', '') for phrase in title_entities if phrase.get('root_lemma')]
                    if any(entity in title_entity_lemmas for entity in query_entities):
                        return row['ID']
                except Exception:
                    pass  # Игнорируем ошибки синтаксического анализа

        # Если точный поиск не дал результатов, используем fuzzy matching
        if RAPIDFUZZ_AVAILABLE and titles_dict:
            # Извлекаем ключевые слова из запроса (убираем стоп-слова)
            stop_words = {'поставь', 'лайк', 'дизлайк', 'на', 'в', 'с', 'и', 'или', 'для', 'что', 'какой', 'какая', 
                         'убери', 'сними', 'удали', 'отмени', 'со', 'спектакля', 'спектакль', 'постановки', 'постановка',
                         'от', 'с', 'к', 'по'}
            query_keywords = [w for w in normalized_query_parts if w not in stop_words and len(w) > 2]
            
            # Сначала пробуем найти по отдельным ключевым словам (для коротких названий типа "Кармен")
            if query_keywords:
                for keyword in query_keywords:
                    # Для очень коротких слов (до 6 символов) используем еще более низкий порог
                    # Для коротких слов (до 8 символов) используем более низкий порог
                    if len(keyword) <= 6:
                        threshold = 60  # Очень низкий порог для очень коротких слов (например, "Кармен" vs "Кормен")
                    elif len(keyword) <= 8:
                        threshold = 65
                    else:
                        threshold = 70
                    matched_title = fuzzy_match(keyword, titles_dict, threshold=threshold)
                    if matched_title:
                        return titles_dict[matched_title]
                
                # Если не нашли по отдельным словам, пробуем всю фразу из ключевых слов
                query_keywords_str = ' '.join(query_keywords)
                if len(query_keywords_str) <= 6:
                    threshold = 60
                elif len(query_keywords_str) <= 10:
                    threshold = 65
                else:
                    threshold = 70
                matched_title = fuzzy_match(query_keywords_str, titles_dict, threshold=threshold)
                if matched_title:
                    return titles_dict[matched_title]
            
            # Если не нашли по ключевым словам, пробуем весь запрос (убираем только основные стоп-слова)
            query_clean = query.lower().strip()
            # Убираем стоп-слова более аккуратно (сохраняем пробелы)
            for stop_word in ['поставь', 'лайк', 'дизлайк', 'убери', 'сними', 'удали', 'отмени', 'со', 'с', 'от']:
                # Убираем слово только если оно отделено пробелами или в начале/конце
                query_clean = re.sub(r'\b' + re.escape(stop_word) + r'\b', '', query_clean)
            query_clean = ' '.join(query_clean.split()).strip()  # Нормализуем пробелы
            
            if query_clean and len(query_clean) >= 3:
                # Для очень коротких запросов используем еще более низкий порог
                if len(query_clean) <= 6:
                    threshold = 60  # Очень низкий порог для очень коротких запросов
                elif len(query_clean) <= 10:
                    threshold = 65
                else:
                    threshold = 70
                matched_title = fuzzy_match(query_clean, titles_dict, threshold=threshold)
                if matched_title:
                    return titles_dict[matched_title]

        return None

    def extract_multiple_ids(self, query, df):
        """Извлекает несколько ID из запроса (для комбинированных запросов) с поддержкой опечаток."""
        ids_found = []
        clean_query = query.lower()

        # Ищем числовые ID
        id_matches = re.findall(r'\b(\d+)\b', query)
        for id_str in id_matches:
            try:
                id_val = int(id_str)
                if id_val in df['ID'].values and id_val not in ids_found:
                    ids_found.append(id_val)
            except:
                pass

        # Ищем по названиям (ищем все возможные совпадения)
        query_lower = query.lower()
        
        # Собираем все названия для fuzzy matching
        titles_dict = {}
        for index, row in df.iterrows():
            title = str(row['Название постановки'])
            titles_dict[title] = row['ID']
        
        # Сначала пробуем точный поиск
        for index, row in df.iterrows():
            title = str(row['Название постановки']).lower()
            title_words = set(title.split())
            query_words = set(query_lower.split())

            # Если название полностью содержится в запросе или есть значительное пересечение
            if title in query_lower or len(title_words.intersection(query_words)) >= 2:
                if row['ID'] not in ids_found:
                    ids_found.append(row['ID'])

        # Если fuzzy matching доступен, ищем дополнительные совпадения с опечатками
        if RAPIDFUZZ_AVAILABLE and titles_dict:
            # Извлекаем потенциальные названия из запроса
            # Разбиваем запрос на части по запятым, точкам и союзам
            query_parts = re.split(r'[,\.]\s*|и\s+|или\s+', query)
            for part in query_parts:
                part = part.strip()
                if not part or len(part) < 3:
                    continue
                
                # Убираем стоп-слова
                stop_words = {'поставь', 'лайк', 'дизлайк', 'на', 'в', 'с', 'и', 'или', 'для', 'что', 'какой', 'какая'}
                part_words = [w for w in part.lower().split() if w not in stop_words and len(w) > 2]
                if not part_words:
                    continue
                
                part_clean = ' '.join(part_words)
                
                # Ищем совпадения с fuzzy matching
                matched_title, score = fuzzy_match_with_score(part_clean, titles_dict, threshold=70)
                if matched_title and titles_dict[matched_title] not in ids_found:
                    ids_found.append(titles_dict[matched_title])

        return ids_found

    def extract_criteria(self, raw_text, available_criteria):
        """Извлекает критерии фильтрации (Упрощенно)."""
        extracted_criteria = {}
        for column_name in available_criteria:
            if column_name.lower() in raw_text:
                extracted_criteria[column_name] = True

        return extracted_criteria

