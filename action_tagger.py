"""
Определение действий пользователя из запросов на естественном языке.
"""

import re
from nlp_analyzers import MorphologicalAnalyzer, SyntacticAnalyzer


class ActionTagger:
    """Определяет основное действие, которое хочет выполнить пользователь."""

    def __init__(self):
        """Инициализация с морфологическим и синтаксическим анализаторами."""
        self.morph_analyzer = MorphologicalAnalyzer()
        self.syntax_analyzer = SyntacticAnalyzer()

    ACTION_MAP = {
        # Лайки
        'лайк': 'ADD_LIKE', 'нравится': 'ADD_LIKE', 'поставь_лайк': 'ADD_LIKE',
        'отметить_нравится': 'ADD_LIKE', 'поставить_лайк': 'ADD_LIKE',
        # Дизлайки
        'дизлайк': 'ADD_DISLIKE', 'не_нравится': 'ADD_DISLIKE',
        'отметить_не_нравится': 'ADD_DISLIKE',
        # Удаление
        'удали': 'REMOVE', 'удалить': 'REMOVE', 'сними': 'REMOVE',
        'снять': 'REMOVE', 'убери': 'REMOVE', 'убрать': 'REMOVE',
        'отмени': 'REMOVE', 'отменить': 'REMOVE',
        # Просмотр
        'покажи': 'DISPLAY', 'показать': 'DISPLAY', 'выведи': 'DISPLAY',
        'вывести': 'DISPLAY', 'список': 'DISPLAY',
        # Сброс
        'сбрось': 'RESET', 'сбросить': 'RESET', 'очисти': 'RESET',
        'очистить': 'RESET', 'обнули': 'RESET', 'обнулить': 'RESET', 'reset': 'RESET',
        # Откат
        'откати': 'UNDO', 'откатить': 'UNDO', 'назад': 'UNDO', 'шаг_назад': 'UNDO',
        # Снятие фильтров
        'сними_фильтр': 'REMOVE_FILTER', 'снять_фильтр': 'REMOVE_FILTER',
        'убери_фильтр': 'REMOVE_FILTER', 'убрать_фильтр': 'REMOVE_FILTER',
        'отмени_фильтр': 'REMOVE_FILTER', 'отменить_фильтр': 'REMOVE_FILTER',
    }

    UNDO_PHRASES = [
        'сделай шаг назад', 'сделай шаг назад', 'сделай шаг на зад',
        'откати на один шаг', 'откати на шаг назад', 'откати на шаг',
        'шаг назад', 'шаг на зад'
    ]
    RESET_PHRASES = [
        'сбрось все фильтры и лайки', 'сбрось все', 'очисти все',
        'очисти систему', 'обнули систему', 'обнули все', 'reset систему',
        'reset all'
    ]
    REMOVE_FILTER_PHRASES = [
        'сними фильтр', 'сними все фильтры', 'убери фильтр', 'убери все фильтры',
        'отмени фильтр', 'отмени фильтрацию', 'убери фильтрацию',
        'сними фильтрацию', 'отключи фильтры'
    ]
    REMOVE_LIKE_PHRASES = [
        'сними лайк', 'удали лайк', 'убери лайк', 'отмени лайк'
    ]
    REMOVE_DISLIKE_PHRASES = [
        'сними дизлайк', 'удали дизлайк', 'убери дизлайк', 'отмени дизлайк'
    ]
    DISPLAY_TABLE_PHRASES = [
        'выведи таблицу текущих спектаклей', 'выведи таблицу спектаклей',
        'покажи таблицу текущих спектаклей', 'покажи таблицу спектаклей',
        'выведи список спектаклей', 'покажи список спектаклей',
        'покажи все спектакли', 'выведи все спектакли'
    ]
    DISPLAY_STATUS_PHRASES = [
        'какое состояние системы', 'покажи состояние системы',
        'покажи статус', 'как работает система сейчас', 'каков статус'
    ]
    DISPLAY_RANKING_PHRASES = [
        'покажи рейтинг', 'выведи рейтинг', 'покажи рекомендации',
        'выведи рекомендации', 'покажи топ', 'выведи топ'
    ]

    # Специальные паттерны для сложных действий
    SPECIAL_PATTERNS = {
        'ADVANCED_SEARCH': [
            'покажи все', 'найди все', 'найди спектакли', 'найди постановки',
            'какие постановки', 'какие спектакли', 'покажи все оперы', 'покажи все мюзиклы'
        ],
        'DISPLAY_RANKING': ['рейтинг', 'рекомендац', 'топ', 'лучших', 'рекомендуете'],
        'DISPLAY_TABLE': ['таблиц', 'список спектакл'],
        'DISPLAY_STATUS': ['статус', 'состояние'],
        'REMOVE_LIKE': ['сними лайк', 'удали лайк', 'убери лайк'],
        'REMOVE_DISLIKE': ['сними дизлайк', 'удали дизлайк', 'убери дизлайк'],
    }

    def tag_action(self, tokenized_text, original_text=None):
        """Ищет главное действие в токенизированном тексте."""
        text_str = " ".join(tokenized_text)
        text_lower = text_str.lower()
        raw_text_lower = original_text.lower() if isinstance(original_text, str) else text_lower

        # Нормализуем текст с помощью морфологического анализа
        normalized_text = self.morph_analyzer.normalize_text(text_str)
        normalized_tokens = normalized_text.split()

        # Используем синтаксический анализ для поиска глаголов и их объектов
        syntax_info = self.syntax_analyzer.find_verb_objects(text_str)

        # 1. Специальные случаи (проверяем первыми, используя исходный текст)
        if any(phrase in raw_text_lower for phrase in self.UNDO_PHRASES):
            return 'UNDO'
        if 'назад' in tokenized_text and len(tokenized_text) <= 2:
            return 'UNDO'
        if any(phrase in raw_text_lower for phrase in self.RESET_PHRASES):
            return 'RESET'
        if any(phrase in raw_text_lower for phrase in self.REMOVE_FILTER_PHRASES):
            return 'REMOVE_FILTER'
        if any(phrase in raw_text_lower for phrase in self.REMOVE_LIKE_PHRASES):
            return 'REMOVE_LIKE'
        if any(phrase in raw_text_lower for phrase in self.REMOVE_DISLIKE_PHRASES):
            return 'REMOVE_DISLIKE'
        if any(phrase in raw_text_lower for phrase in self.DISPLAY_TABLE_PHRASES):
            return 'DISPLAY_TABLE'
        if any(phrase in raw_text_lower for phrase in self.DISPLAY_STATUS_PHRASES):
            return 'DISPLAY_STATUS'
        if any(phrase in raw_text_lower for phrase in self.DISPLAY_RANKING_PHRASES):
            return 'DISPLAY_RANKING'

        # 2. Проверка специальных паттернов
        for action_type, patterns in self.SPECIAL_PATTERNS.items():
            for pattern in patterns:
                if pattern in raw_text_lower:
                    # Дополнительные проверки для некоторых действий
                    if action_type == 'ADVANCED_SEARCH':
                        # Проверяем наличие критериев поиска
                        if any(w in raw_text_lower for w in ['жанр', 'рейтинг', 'театр', 'стиль', 'длительность',
                                                          'оперы', 'мюзиклы', 'балет', 'комедия', 'выше', 'ниже',
                                                          'больше', 'меньше', 'в жанре', 'с рейтингом']):
                            return 'ADVANCED_SEARCH'
                    elif action_type in ['REMOVE_LIKE', 'REMOVE_DISLIKE']:
                        return action_type
                    elif action_type == 'DISPLAY_TABLE':
                        if 'спектакл' in raw_text_lower and any(w in raw_text_lower for w in ['покажи', 'выведи', 'список']):
                            return 'DISPLAY_TABLE'
                    elif action_type == 'DISPLAY_RANKING':
                        # Проверяем, что это действительно запрос на рекомендации, а не просто упоминание слова
                        if any(w in raw_text_lower for w in ['рейтинг', 'рекомендац', 'топ', 'лучших', 'рекомендуете']):
                            return 'DISPLAY_RANKING'
                    else:
                        return action_type

        # 2.5. Использование синтаксического анализа для поиска действий
        if syntax_info:
            for obj_info in syntax_info:
                verb_lemma = obj_info['verb']
                # Проверяем нормализованные формы глаголов
                if verb_lemma in ['показать', 'показывать', 'вывести', 'выводить']:
                    return 'DISPLAY'
                elif verb_lemma in ['найти', 'искать', 'находить']:
                    return 'ADVANCED_SEARCH'
                elif verb_lemma in ['поставить', 'ставить', 'добавить', 'добавлять']:
                    if 'лайк' in text_lower or 'нравиться' in text_lower:
                        return 'ADD_LIKE'
                    elif 'дизлайк' in text_lower or 'не_нравиться' in text_lower:
                        return 'ADD_DISLIKE'

        # 3. Поиск в ACTION_MAP (используем нормализованные токены)
        for token in normalized_tokens:
            token_lower = token.lower()
            if token_lower in self.ACTION_MAP:
                action = self.ACTION_MAP[token_lower]
                # Дополнительная обработка для некоторых действий
                if action == 'REMOVE':
                    if 'лайк' in raw_text_lower:
                        return 'REMOVE_LIKE'
                    elif 'дизлайк' in raw_text_lower:
                        return 'REMOVE_DISLIKE'
                elif action == 'DISPLAY':
                    if 'рейтинг' in raw_text_lower or 'рекомендац' in raw_text_lower:
                        return 'DISPLAY_RANKING'
                    elif 'таблиц' in raw_text_lower:
                        return 'DISPLAY_TABLE'
                    elif 'статус' in raw_text_lower or 'состояние' in raw_text_lower:
                        return 'DISPLAY_STATUS'
                return action

        # 4. Лайки/Дизлайки (проверяем по контексту, если не нашли в ACTION_MAP)
        is_like = 'лайк' in raw_text_lower or 'нравится' in raw_text_lower
        is_dislike = 'дизлайк' in raw_text_lower or 'не_нравится' in raw_text_lower or 'не нравится' in raw_text_lower
        is_add_command = any(w in raw_text_lower for w in ['поставь', 'поставить', 'добавь', 'добавить', 'отметь', 'отметить'])
        has_id = bool(re.search(r'\d+', text_str))

        # Проверяем на удаление перед добавлением
        if any(w in raw_text_lower for w in ['удали', 'удалить', 'сними', 'снять', 'убери', 'убрать', 'отмени', 'отменить']):
            if is_like:
                return 'REMOVE_LIKE'
            elif is_dislike:
                return 'REMOVE_DISLIKE'

        if is_like and (is_add_command or has_id):
            return 'ADD_LIKE'
        if is_dislike and (is_add_command or has_id):
            return 'ADD_DISLIKE'

        # 5. Проверка на снятие фильтров
        if any(w in raw_text_lower for w in ['сними фильтр', 'убери фильтр', 'отмени фильтр', 'сними все фильтры']):
            return 'REMOVE_FILTER'

        return 'UNKNOWN'

