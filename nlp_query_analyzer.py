"""
Анализатор запросов на естественном языке.
Координирует работу различных NLP компонентов.
"""

import re
from action_tagger import ActionTagger
from entity_extractor import EntityExtractor
from query_to_filter import QueryToFilterConverter
from nlp_analyzers import MorphologicalAnalyzer, SyntacticAnalyzer


class NLPQueryAnalyzer:
    """Координирует работу анализаторов и готовит команду для системы."""

    def __init__(self, df_columns, initial_df, duration_linguistic=None):
        self.action_tagger = ActionTagger()
        self.entity_extractor = EntityExtractor()
        self.filter_converter = QueryToFilterConverter(df_columns, duration_linguistic)
        self.available_criteria = [
            'Целевая аудитория', 'Жанр', 'Язык исполнения',
            'Театр', 'Стиль', 'Режиссер', 'Рейтинг', 'Длительность'
        ]
        self.initial_df = initial_df
        self.morph_analyzer = MorphologicalAnalyzer()
        self.syntax_analyzer = SyntacticAnalyzer()

    def analyze(self, query, current_df):
        """Основной метод анализа запроса, расширенный для всех типов команд."""

        self.entity_extractor.current_df = current_df

        clean_query = query.lower().replace('.', '').replace(',', '').strip()
        clean_query = clean_query.replace('не нравится', 'не_нравится')

        normalized_query = self.morph_analyzer.normalize_text(clean_query)
        tokenized = normalized_query.split()

        syntax_doc = self.syntax_analyzer.analyze(query)

        action = self.action_tagger.tag_action(tokenized, query)
        entity_id = self.entity_extractor.extract_id(tokenized, clean_query)

        has_genre_keyword = any(genre in clean_query for genre in ['оперы', 'опера', 'мюзиклы', 'мюзикл', 'балет', 'комедии', 'комедия', 'театр'])
        has_search_keyword = any(w in clean_query for w in ['покажи', 'найди', 'какие', 'все', 'выведи'])
        
        has_duration_linguistic_term = False
        if self.filter_converter.duration_linguistic:
            term, _ = self.filter_converter.duration_linguistic.find_term_in_query(query)
            if term:
                has_duration_linguistic_term = True
        
        if has_search_keyword and (has_genre_keyword or has_duration_linguistic_term):
            filters = self.filter_converter.get_query_parts(query, self.initial_df)
            if isinstance(filters, list) and len(filters) > 0:
                try:
                    if any(f.get('column') == 'Жанр' for f in filters) or \
                       any(f.get('column') == 'Длительность (мин)' for f in filters):
                        action = 'ADVANCED_SEARCH'
                        entity_id = None
                except Exception:
                    pass

        if entity_id is None and action != 'ADVANCED_SEARCH':
            entity_id = self.entity_extractor.find_id_by_title(clean_query, current_df)
            
            if entity_id is not None and action != 'ADVANCED_SEARCH' and (action == 'UNKNOWN' or action in ['ADD_LIKE', 'ADD_DISLIKE', 'REMOVE_LIKE', 'REMOVE_DISLIKE']):
                has_like_keywords = any(w in clean_query for w in ['лайк', 'нравится', 'поставь', 'добавь', 'отметь'])
                has_dislike_keywords = 'дизлайк' in clean_query or 'не нравится' in clean_query or 'не_нравится' in clean_query
                has_remove_keywords = any(w in clean_query for w in ['убери', 'сними', 'удали', 'отмени'])
                
                if has_remove_keywords:
                    if 'лайк' in clean_query:
                        action = 'REMOVE_LIKE'
                    elif 'дизлайк' in clean_query or has_dislike_keywords:
                        action = 'REMOVE_DISLIKE'
                elif has_like_keywords or has_dislike_keywords:
                    if has_dislike_keywords:
                        action = 'ADD_DISLIKE'
                    elif has_like_keywords:
                        action = 'ADD_LIKE'

        structured_command = {
            'action': action,
            'id': entity_id,
            'query': query,
            'params': {}
        }
        
        if action == 'ADVANCED_SEARCH':
            filters = self.filter_converter.get_query_parts(query, self.initial_df)
            if isinstance(filters, list) and len(filters) > 0:
                structured_command['params']['filters'] = filters

        if any(w in clean_query for w in ['самый высокий', 'самый низкий', 'максимальный', 'минимальный',
                                          'какой театр имеет', 'какой спектакль имеет', 'какая постановка имеет',
                                          'покажи спектакли с максимальным', 'покажи спектакли с минимальным',
                                          'покажи постановки с максимальным', 'покажи постановки с минимальным']):
            structured_command['action'] = 'AGGREGATION'
            structured_command['params']['filters'] = self.filter_converter.get_query_parts(query, self.initial_df)

        elif action == 'UNKNOWN' or action == 'ADVANCED_SEARCH' or action == 'DISPLAY_RANKING' or \
             (action == 'DISPLAY' and (has_genre_keyword or has_duration_linguistic_term)):

            if action == 'DISPLAY_RANKING' or re.search(r'(топ|лучших|наиболее|дай.*рекомендац)\s*(\d+)', clean_query) or 'рекомендуете' in clean_query or ('дай' in clean_query and 'рекомендац' in clean_query):
                structured_command['action'] = 'DISPLAY_RANKING'
                match = re.search(r'(\d+)', clean_query)
                structured_command['params']['top_n'] = int(match.group(1)) if match else 5

            elif action == 'ADVANCED_SEARCH' or has_duration_linguistic_term or any(w in clean_query for w in [
                'какие', 'найди все', 'найди спектакли', 'найди постановки',
                'покажи все оперы', 'покажи все мюзиклы', 'покажи все спектакли',
                'покажи все', 'с рейтингом', 'где', 'в жанре', 'идут в',
                'какие постановки', 'какие спектакли', 'постановки с', 'спектакли с',
                'покажи оперы', 'покажи мюзиклы', 'покажи балет', 'покажи комедии',
                'выведи все', 'выведи спектакли', 'выведи постановки'
            ]) or ('покажи' in clean_query and any(genre in clean_query for genre in ['оперы', 'опера', 'мюзиклы', 'мюзикл', 'балет', 'комедии', 'комедия'])):
                filters = self.filter_converter.get_query_parts(query, self.initial_df)
                if filters:
                    structured_command['action'] = 'ADVANCED_SEARCH'
                    structured_command['params']['filters'] = filters
                elif any(w in clean_query for w in ['покажи все', 'покажи спектакли', 'покажи постановки', 'выведи все', 'выведи спектакли', 'выведи постановки']):
                    structured_command['action'] = 'DISPLAY_TABLE'

            elif any(w in clean_query for w in ['как мне начать', 'как начать', 'с чего начать', 'начать пользоваться',
                                                 'объясните', 'каковы основные правила', 'что произойдет', 'вес',
                                                 'как работают', 'как работает']):
                structured_command['action'] = 'GENERAL_QUESTION'

            has_like_dislike = 'лайк' in clean_query or 'дизлайк' in clean_query
            has_recommendation_request = ('рекомендуете' in clean_query or 'порекомендуете' in clean_query or
                                         'порекомендуй' in clean_query or 'рекомендац' in clean_query or
                                         (clean_query.endswith('что') or ' что' in clean_query))

            if has_like_dislike and has_recommendation_request:
                structured_command['action'] = 'LIKE_DISLIKE_AND_RANK'

            elif self.entity_extractor.extract_criteria(clean_query, self.available_criteria):
                filters = self.filter_converter.get_query_parts(query, self.initial_df)
                if filters:
                    structured_command['action'] = 'ADVANCED_SEARCH'
                    structured_command['params']['filters'] = filters

        if structured_command['action'] in ['ADD_LIKE', 'ADD_DISLIKE', 'REMOVE_LIKE', 'REMOVE_DISLIKE', 'LIKE_DISLIKE_AND_RANK'] and structured_command['id'] is None:
            if structured_command['action'] != 'LIKE_DISLIKE_AND_RANK':
                 structured_command['action'] = 'UNKNOWN'
        
        if structured_command['id'] is not None:
            has_like_keywords = any(w in clean_query for w in ['лайк', 'нравится', 'поставь', 'добавь', 'отметь'])
            has_dislike_keywords = 'дизлайк' in clean_query or 'не нравится' in clean_query or 'не_нравится' in clean_query
            has_remove_keywords = any(w in clean_query for w in ['убери', 'сними', 'удали', 'отмени'])
            
            if structured_command['action'] == 'UNKNOWN' or structured_command['action'] in ['ADD_LIKE', 'ADD_DISLIKE', 'REMOVE_LIKE', 'REMOVE_DISLIKE']:
                if has_remove_keywords:
                    if 'лайк' in clean_query:
                        structured_command['action'] = 'REMOVE_LIKE'
                    elif 'дизлайк' in clean_query or has_dislike_keywords:
                        structured_command['action'] = 'REMOVE_DISLIKE'
                elif has_like_keywords or has_dislike_keywords:
                    if has_dislike_keywords:
                        structured_command['action'] = 'ADD_DISLIKE'
                    elif has_like_keywords:
                        structured_command['action'] = 'ADD_LIKE'

        return structured_command
