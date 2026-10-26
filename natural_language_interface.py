"""
Интерфейс для работы с системой на естественном языке.
"""

from recommendation_system import InteractiveTreeRecommendationSystem
from nlp_query_analyzer import NLPQueryAnalyzer


class NaturalLanguageInterface:

    def __init__(self, df):
        self.system = InteractiveTreeRecommendationSystem(df)
        self.analyzer = NLPQueryAnalyzer(df.columns, df, self.system.duration_linguistic)
        self.available_filter_columns = ['Целевая аудитория', 'Жанр', 'Язык исполнения', 'Театр', 'Стиль', 'Режиссер']

    def show_help(self):
        print("\n🎮 ДОСТУПНЫЕ КОМАНДЫ НА ЕСТЕСТВЕННОМ ЯЗЫКЕ:")
        print("--- Управление Лайками ---")
        print(" * **Поставь лайк ID 3.** / **Поставь лайк Кармен**")
        print(" * **Сними дизлайк со спектакля 8.**")
        print("--- Откат/Сброс ---")
        print(" * **Сделай шаг назад.** / **Откати на один шаг.**")
        print(" * **Сбрось все фильтры и лайки.**")
        print("--- Фильтрация/Поиск ---")
        print(" * **Покажи все оперы и мюзиклы.**")
        print(" * **Найди все спектакли, где Рейтинг выше 8.0 и Жанр Опера.** (Сложный поиск)")
        print(" * **Покажи все не очень длинные постановки.** (Лингвистическая переменная)")
        print(" * **Найди длинные спектакли.** / **Покажи недлинные постановки.**")
        print(" * **Какой спектакль имеет самый высокий рейтинг?** (Агрегация)")
        print(" * **Покажи топ-5 рекомендаций.** / **Покажи рейтинг.**")
        print(" * **Хочу отфильтровать по жанру и театру.** (Старый интерактивный режим)")
        print(" * **Выведи таблицу текущих спектаклей.** / **Какое состояние системы?**")
        print(" * **Выйти.**")

    def process_command(self, query):
        """Обрабатывает структурированную команду."""

        command = self.analyzer.analyze(query, self.system.current_df)
        action = command['action']
        entity_id = command['id']
        params = command['params']

        print(f"\n[Анализ] -> Действие: **{action}**, ID: **{entity_id}**")
        print("-" * 50)

        if action == 'ADD_LIKE':
            self.system.add_like(entity_id)
        elif action == 'ADD_DISLIKE':
            self.system.add_dislike(entity_id)
        elif action == 'REMOVE_LIKE':
            self.system.remove_like(entity_id)
        elif action == 'REMOVE_DISLIKE':
            self.system.remove_dislike(entity_id)
        elif action == 'UNDO':
            self.system.undo_action()
        elif action == 'RESET':
            self.system.reset_system()
        elif action == 'DISPLAY_RANKING':
            self.system.show_ranking(top_n=params.get('top_n'))
        elif action == 'DISPLAY_TABLE':
            self.system.show_spectacles_table()
        elif action == 'DISPLAY_STATUS':
            self.system.show_status()
        elif action == 'ADVANCED_SEARCH':
            self.system._apply_natural_filter(params['filters'])
        elif action == 'AGGREGATION':
            self.system.get_aggregation_result(query, params.get('filters'))
        elif action == 'GENERAL_QUESTION':
            self.system.answer_general_question(query)
        elif action == 'LIKE_DISLIKE_AND_RANK':
            self.system.process_like_dislike_and_rank(query, self.system.current_df, self.analyzer.entity_extractor)
        elif action == 'REMOVE_FILTER':
            self.system.remove_filter()
        elif action == 'UNKNOWN':
            query_lower = query.lower()
            if any(w in query_lower for w in ['покажи спектакл', 'выведи таблиц', 'покажи таблиц', 'список спектакл', 'покажи все спектакл']):
                self.system.show_spectacles_table()
            elif any(w in query_lower for w in ['с чего начать', 'как начать', 'начать пользоваться']):
                self.system.answer_general_question(query)
            elif ('лайк' in query_lower or 'дизлайк' in query_lower) and 'что' in query_lower:
                self.system.process_like_dislike_and_rank(query, self.system.current_df, self.analyzer.entity_extractor)
            else:
                print("❌ Прошу прощения, я не могу ответить на ваш вопрос.")
        else:
             print(f"⚠️ Необработанное действие: {action}")

        self.analyzer = NLPQueryAnalyzer(self.system.initial_df.columns, self.system.initial_df, self.system.duration_linguistic)

    def run(self):
        """Основной цикл для интерактивного режима."""
        self.show_help()
        while True:
            try:
                user_input = input("\n💬 Ваш запрос: ").strip()
                if user_input.lower() in ['выйти', 'выход', 'quit']:
                    print("👋 Завершение работы.")
                    break
                if user_input.lower() in ['помощь', 'help']:
                    self.show_help()
                    continue

                self.process_command(user_input)

            except EOFError:
                print("\n👋 Завершение работы.")
                break
            except Exception as e:
                print(f"\n🚫 ПРОИЗОШЛА КРИТИЧЕСКАЯ ОШИБКА: {e}")

