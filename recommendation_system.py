import re
from collections import deque

import numpy as np
import pandas as pd
from tabulate import tabulate

from criteria_tree import CriteriaTree
from linguistic_variables import LinguisticVariable


class InteractiveTreeRecommendationSystem:
    """Интерактивная система рекомендаций с фильтрацией по дереву и лайками"""

    # === ⭐️ МАТРИЦЫ НЕЕВКЛИДОВОЙ СХОЖЕСТИ (Без изменений) ===
    GENRE_SIMILARITY = {
        ('Балет', 'Опера'): 0.8, ('Театр', 'Опера'): 0.85,
        ('Театр', 'Балет'): 0.65, ('Мьюзикл', 'Театр'): 0.7,
        ('Комедия', 'Опера'): 0.2, ('Иммерсивный театр', 'Театр'): 0.9,
        ('Семейный', 'Комедия'): 0.7,
    }
    STYLE_SIMILARITY = {
        ('Классика', 'Классическая опера'): 0.9, ('Классика', 'Драма'): 0.85,
        ('Классика', 'Трагедия'): 0.8, ('Современная опера', 'Лирическая опера'): 0.7,
        ('Психологическая драма', 'Драма'): 0.95, ('Классика', 'Шоу'): 0.3,
        ('Современный балет', 'Романтический балет'): 0.75,
    }
    THEATRE_SIMILARITY = {
        ('Театр им. Пушкина', 'Музыкальный театр им. К.С. Станиславского и Вл.И. Немировича-Данченко.'): 0.7,
    }
    LANGUAGE_SIMILARITY = {
        ('Русский', 'Французский'): 0.65,
        ('Русский', 'Итальянский'): 0.65,
        ('Французский', 'Итальянский'): 0.9,
    }
    AUDIENCE_SIMILARITY = {
        ('Взрослая', 'Семейная'): 0.85,
        ('Семейная', 'Для детей'): 0.9,
        ('Взрослая', 'Для детей'): 0.3,
    }
    FEATURE_WEIGHTS = {
        'Жанр': 0.25, 'Рейтинг': 0.25, 'Стиль': 0.15, 'Театр': 0.10,
        'Целевая аудитория': 0.10, 'Длительность (мин)': 0.05, 'Язык исполнения': 0.05,
    }
    BINARY_COLUMNS = [
        'Проводится на открытом воздухе?', 'Семейный?', 'Для детей?', 'Платный?'
    ]

    def __init__(self, df):
        self.initial_df = df.copy() # Переименовано original_df в initial_df для ясности
        # Предобработка
        if self.initial_df['Рейтинг'].dtype == object:
             self.initial_df['Рейтинг'] = self.initial_df['Рейтинг'].astype(str).str.replace(',', '.', regex=False)
        self.initial_df['Рейтинг'] = pd.to_numeric(self.initial_df['Рейтинг'], errors='coerce')

        self.max_duration = self.initial_df['Длительность (мин)'].max()
        self.min_duration = self.initial_df['Длительность (мин)'].min()
        self.max_rating = self.initial_df['Рейтинг'].max()
        self.min_rating = self.initial_df['Рейтинг'].min()

        self.current_df = self.initial_df.copy()
        self.likes = []
        self.dislikes = []
        self.ranking = None
        self.filter_criteria = {}

        # --- Добавление для UNDO/Отката ---
        self.action_history = deque(maxlen=20) # Очередь для хранения последних 20 действий
        self._save_state() # Сохраняем начальное состояние

        # --- Инициализация CriteriaTree для фильтрации ---
        filter_columns = ['Жанр', 'Театр', 'Стиль', 'Язык исполнения', 'Целевая аудитория']
        self.criteria_tree = CriteriaTree(self.initial_df, filter_columns)
        
        # --- Инициализация лингвистической переменной для длительности ---
        self.duration_linguistic = LinguisticVariable('Длительность', self.min_duration, self.max_duration)
        print(f"📊 Лингвистическая переменная 'Длительность' инициализирована:")
        print(f"   Диапазон: {self.min_duration:.1f} - {self.max_duration:.1f} минут")
        print(f"   Доступные термы: {', '.join(self.duration_linguistic.terms.keys())}")

        print("🚀 ИНТЕРАКТИВНАЯ СИСТЕМА С ФИЛЬТРАЦИЕЙ И МНОГОМЕРНЫМИ МЕТРИКАМИ")
        print("=" * 70)
        print(f"📊 Загружено {len(df)} спектаклей")

    # === ВСПОМОГАТЕЛЬНЫЕ МЕТОДЫ ДЛЯ МЕТРИК (Без изменений) ===
    def _get_similarity(self, val1, val2, similarity_map):
        if pd.isna(val1) or pd.isna(val2): return 0.5
        if val1 == val2: return 1.0
        key = tuple(sorted((val1, val2)))
        return similarity_map.get(key, 0.5)

    def _calculate_genre_similarity(self, genre1, genre2): return self._get_similarity(genre1, genre2, self.GENRE_SIMILARITY)
    def _calculate_style_similarity(self, style1, style2): return self._get_similarity(style1, style2, self.STYLE_SIMILARITY)
    def _calculate_theatre_similarity(self, theatre1, theatre2): return self._get_similarity(theatre1, theatre2, self.THEATRE_SIMILARITY)
    def _calculate_language_similarity(self, lang1, lang2): return self._get_similarity(lang1, lang2, self.LANGUAGE_SIMILARITY)
    def _calculate_audience_similarity(self, aud1, aud2): return self._get_similarity(aud1, aud2, self.AUDIENCE_SIMILARITY)

    def _calculate_numeric_similarity(self, val1, val2, min_val, max_val):
        if pd.isna(val1) or pd.isna(val2): return 0.5
        range_val = max_val - min_val
        if range_val <= 0 or range_val == np.inf: return 1.0
        distance = np.sqrt((val1 - val2)**2)
        norm_distance = distance / range_val
        return 1.0 - norm_distance

    def _calculate_binary_similarity(self, val1, val2):
        if pd.isna(val1) or pd.isna(val2): return 0.5
        v1 = 1 if str(val1).lower() == 'да' else 0
        v2 = 1 if str(val2).lower() == 'да' else 0
        return 1.0 if v1 == v2 else 0.0

    # --- МЕТОД ОБНОВЛЕНИЯ РЕЙТИНГА (Без изменений) ---
    def _update_ranking(self, show_ranking=True):
        if not self.likes and not self.dislikes:
            self.ranking = None
            return

        ranking_df = self.current_df.copy().reset_index(drop=True)
        current_data = self.current_df.reset_index(drop=True)
        similarity_scores = np.zeros(len(ranking_df))

        binary_weight = 0.05 / len(self.BINARY_COLUMNS)

        def calculate_object_similarity(current_row, target_row):
            sim_genre = self._calculate_genre_similarity(current_row.get('Жанр'), target_row.get('Жанр'))
            sim_style = self._calculate_style_similarity(current_row.get('Стиль'), target_row.get('Стиль'))
            sim_theatre = self._calculate_theatre_similarity(current_row.get('Театр'), target_row.get('Театр'))
            sim_language = self._calculate_language_similarity(current_row.get('Язык исполнения'), target_row.get('Язык исполнения'))
            sim_audience = self._calculate_audience_similarity(current_row.get('Целевая аудитория'), target_row.get('Целевая аудитория'))
            sim_rating = self._calculate_numeric_similarity(current_row.get('Рейтинг'), target_row.get('Рейтинг'), self.min_rating, self.max_rating)
            sim_duration = self._calculate_numeric_similarity(current_row.get('Длительность (мин)'), target_row.get('Длительность (мин)'), self.min_duration, self.max_duration)

            sim_binary = 0
            for col in self.BINARY_COLUMNS:
                sim_binary += self._calculate_binary_similarity(current_row.get(col), target_row.get(col)) * binary_weight

            feature_similarity = (
                sim_genre * self.FEATURE_WEIGHTS.get('Жанр', 0.25) +
                sim_rating * self.FEATURE_WEIGHTS.get('Рейтинг', 0.25) +
                sim_style * self.FEATURE_WEIGHTS.get('Стиль', 0.15) +
                sim_theatre * self.FEATURE_WEIGHTS.get('Театр', 0.10) +
                sim_audience * self.FEATURE_WEIGHTS.get('Целевая аудитория', 0.10) +
                sim_duration * self.FEATURE_WEIGHTS.get('Длительность (мин)', 0.05) +
                sim_language * self.FEATURE_WEIGHTS.get('Язык исполнения', 0.05) +
                sim_binary
            )
            return feature_similarity

        DISLIKE_PENALTY_FACTOR = 0.5
        for idx, row in ranking_df.iterrows():
            total_similarity = 0
            for like_id in self.likes:
                like_row = current_data[current_data['ID'] == like_id]
                if not like_row.empty:
                    total_similarity += calculate_object_similarity(row, like_row.iloc[0])

            for dislike_id in self.dislikes:
                dislike_row = current_data[current_data['ID'] == dislike_id]
                if not dislike_row.empty:
                    total_similarity -= calculate_object_similarity(row, dislike_row.iloc[0]) * DISLIKE_PENALTY_FACTOR

            normalization_factor = len(self.likes) + len(self.dislikes) * DISLIKE_PENALTY_FACTOR
            if normalization_factor > 0:
                similarity_scores[idx] = total_similarity / normalization_factor
            else:
                similarity_scores[idx] = total_similarity

        ranking_df['Similarity_Score'] = similarity_scores

        if not self.likes and self.dislikes:
            ranking_df['Rank'] = ranking_df['Similarity_Score'].rank(ascending=True, method='min')
        else:
            ranking_df['Rank'] = ranking_df['Similarity_Score'].rank(ascending=False, method='min')

        recommendation_df = ranking_df[
            (~ranking_df['ID'].isin(self.likes)) &
            (~ranking_df['ID'].isin(self.dislikes))
        ].copy()

        self.ranking = recommendation_df.sort_values('Similarity_Score', ascending=False)
        if show_ranking:
            print("✅ Рейтинг обновлен!")
            print(f"💡 Создано {len(self.ranking)} рекомендаций")
            self.show_ranking()

    # --- МЕТОДЫ ДЛЯ UNDO/ОТКАТА (Новые) ---

    def _save_state(self, action_type="INIT"):
        """Сохраняет текущее состояние системы в историю."""
        state = {
            'action': action_type,
            'likes': self.likes.copy(),
            'dislikes': self.dislikes.copy(),
            'current_df': self.current_df.copy(),
            'filter_criteria': self.filter_criteria.copy()
        }
        self.action_history.append(state)

    def undo_action(self):
        """Откатывает последнее действие, восстанавливая предыдущее состояние (на 1 шаг назад)."""
        if len(self.action_history) <= 1:
            print("⚠️ Нет действий для отката или вы находитесь в начальном состоянии.")
            return

        # Удаляем ТЕКУЩЕЕ состояние (последнее в очереди - состояние ПОСЛЕ последнего действия)
        current_state = self.action_history.pop()

        # Если текущее состояние начинается с "AFTER_", значит это состояние ПОСЛЕ действия
        # Нужно найти состояние ДО этого действия (которое было сохранено ПЕРЕД действием)
        if current_state['action'].startswith('AFTER_'):
            # Удаляем состояние ПОСЛЕ действия
            # Теперь последнее в очереди - это состояние ДО действия (которое мы сохранили перед действием)
            if len(self.action_history) > 0:
                previous_state = self.action_history[-1]
            else:
                print("⚠️ Не найдено предыдущее состояние для отката.")
                return
        else:
            # Если состояние не начинается с "AFTER_", значит это старое поведение
            # Просто берем предыдущее состояние
            if len(self.action_history) > 0:
                previous_state = self.action_history[-1]
            else:
                print("⚠️ Не найдено предыдущее состояние для отката.")
                return

        # Восстанавливаем предыдущее состояние
        self.likes = previous_state['likes'].copy()
        self.dislikes = previous_state['dislikes'].copy()
        self.current_df = previous_state['current_df'].copy()
        self.filter_criteria = previous_state['filter_criteria'].copy()
        self.ranking = None # Обнуляем рейтинг

        # Дополнительная проверка на сброс фильтров
        if not self.filter_criteria and len(self.current_df) != len(self.initial_df):
            self.current_df = self.initial_df.copy()

        # Для корректного вывода: если это не INIT, берем метку действия из предыдущего состояния
        action_label = previous_state['action']
        # Убираем префикс "AFTER_" для более понятного вывода
        if action_label.startswith('AFTER_'):
            action_label = action_label.replace('AFTER_', '')

        if action_label == 'INIT':
            print(f"↩️ Система откачена до **начального состояния**.")
        else:
            print(f"↩️ Система откачена до состояния после действия: **{action_label}**.")

        print(f"📊 Текущее количество спектаклей: {len(self.current_df)}")

        # ВАЖНО: НЕ сохраняем состояние при откате, иначе следующий откат не сработает
        # Просто обновляем рейтинг, если в восстановленном состоянии есть лайки/дизлайки
        if self.likes or self.dislikes:
            self._update_ranking()
        else:
            self.show_status()

    # === НОВЫЙ МЕТОД: ФИЛЬТРАЦИЯ НА ЕСТЕСТВЕННОМ ЯЗЫКЕ ===
    def _apply_natural_filter(self, filters):
        """
        Применяет фильтры, полученные из NLP-запроса, к исходному DataFrame.
        Использует CriteriaTree для получения доступных значений.
        :param filters: Список словарей с критериями фильтрации.
        """
        if not filters:
            print("❌ Не удалось извлечь критерии фильтрации из запроса.")
            return False
        
        # Убеждаемся, что filters это список
        if not isinstance(filters, list):
            print(f"❌ Неверный тип filters: {type(filters)}")
            return False

        self._save_state(action_type="ADVANCED_SEARCH_FILTER") # Сохраняем перед фильтрацией

        df = self.initial_df.copy() # Всегда начинаем с полного набора данных
        current_criteria = {}

        for filter_item in filters:
            col = filter_item['column']
            op = filter_item['operator']
            val = filter_item['value']
            
            # Логика для числовых колонок
            if col in ['Рейтинг', 'Длительность (мин)']:
                try:
                    # Проверяем, не является ли это диапазоном (для лингвистических переменных)
                    # Если это диапазон, он уже должен быть разбит на два фильтра
                    # Но на всякий случай проверяем
                    if op in ['>=', '<=', '>', '<', '==']:
                        df = df.query(f"`{col}` {op} {val}")
                        # Обновляем критерий, добавляя к существующему, если есть
                        if col in current_criteria:
                            current_criteria[col] += f" и {op} {val}"
                        else:
                            current_criteria[col] = f"{op} {val}"
                    else:
                        print(f"⚠️ Неизвестный оператор для числового фильтра: {op}")
                        continue
                except Exception as e:
                    print(f"⚠️ Ошибка при применении числового фильтра по {col}: {e}")
                    continue

            # Логика для строковых/бинарных колонок
            # Используем CriteriaTree для проверки доступных значений
            else:
                if op == '==':
                    # Убеждаемся, что val - это скалярное значение, а не массив
                    if isinstance(val, (list, tuple)):
                        if len(val) == 0:
                            continue
                        val = val[0]  # Берем первое значение, если это список/кортеж
                    elif isinstance(val, np.ndarray):
                        if val.size == 0:
                            continue
                        val = val.item() if val.size == 1 else val[0]  # Преобразуем в скаляр
                    elif isinstance(val, pd.Series):
                        if len(val) == 0:
                            continue
                        val = val.iloc[0]  # Берем первое значение из Series
                    
                    # Преобразуем в строку, если это не число
                    if not isinstance(val, (int, float)):
                        val = str(val)
                    
                    # Проверяем, есть ли такое значение в дереве критериев
                    available_values = self.criteria_tree.get_children_values("Root")
                    if col == 'Жанр':
                        # Получаем доступные жанры из первого уровня дерева
                        genre_node = None
                        for child in self.criteria_tree.root.children:
                            if str(child.value) == str(val):
                                genre_node = child
                                break
                        if genre_node is None:
                            print(f"⚠️ Значение '{val}' не найдено в доступных значениях для {col}")
                            continue
                    
                    # Применяем фильтр безопасно
                    try:
                        # Убеждаемся, что df[col] это Series
                        col_series = df[col]
                        
                        if not isinstance(col_series, pd.Series):
                            col_series = pd.Series(col_series, index=df.index)
                        
                        # Убеждаемся, что сравниваем строки со строками или числа с числами
                        if col_series.dtype == 'object':  # Строковая колонка
                            # Преобразуем в строки и сравниваем
                            mask = col_series.astype(str) == str(val)
                        else:  # Числовая колонка
                            if isinstance(val, str):
                                try:
                                    val = float(val) if '.' in str(val) else int(val)
                                except ValueError:
                                    # Если не удалось преобразовать в число, сравниваем как строки
                                    mask = col_series.astype(str) == str(val)
                                else:
                                    mask = col_series == val
                            else:
                                mask = col_series == val
                        
                        # Убеждаемся, что mask - это Series, а не массив
                        if isinstance(mask, np.ndarray):
                            mask = pd.Series(mask, index=df.index)
                        elif not isinstance(mask, pd.Series):
                            # Если mask это что-то другое, создаем Series
                            mask = pd.Series(mask, index=df.index)
                        
                        # Проверяем, что mask не пустой
                        if len(mask) == 0:
                            continue
                        
                        # Проверяем any() безопасно
                        try:
                            has_any = mask.any()
                            if not has_any:
                                continue
                        except Exception as e:
                            # Если any() не работает, проверяем вручную
                            if not any(mask):
                                continue
                        
                        # Применяем фильтр
                        df = df.loc[mask]
                        current_criteria[col] = f"== {val}"
                    except Exception as e:
                        print(f"⚠️ Ошибка при применении фильтра по {col}: {e}")
                        import traceback
                        traceback.print_exc()
                        continue
                # Добавить другую логику, если нужно (например, LIKE)

        # Обновление состояния
        self.current_df = df
        self.filter_criteria = current_criteria
        self.ranking = None

        # Фильтрация лайков/дизлайков (оставляем только те, что в новом df)
        current_ids = self.current_df['ID'].values
        self.likes = [like for like in self.likes if like in current_ids]
        self.dislikes = [dislike for dislike in self.dislikes if dislike in current_ids]

        print("🌳 ФИЛЬТРАЦИЯ ПРИМЕНЕНА (с использованием CriteriaTree)")
        for key, val in self.filter_criteria.items():
            print(f"✅ Фильтр по **{key}**: {val}")
        print(f"💡 Найдено {len(self.current_df)} записей.")

        if len(self.current_df) > 0:
          print(f"\n🎭 ОТФИЛЬТРОВАННЫЕ СПЕКТАКЛИ:")
          display_df = self.current_df[['ID', 'Название постановки', 'Жанр', 'Рейтинг', 'Длительность (мин)', 'Театр']]
          print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.1f'))
        else:
            print("⚠️ Нет спектаклей, соответствующих критериям фильтрации")

        if self.likes or self.dislikes:
            self._update_ranking()
        else:
            self.show_status()

        return True

    def get_available_filter_values(self, column_name, parent_value=None):
        """Получает доступные значения для фильтрации из CriteriaTree."""
        if parent_value is None:
            # Получаем значения первого уровня (по первой колонке в дереве)
            if self.criteria_tree.root.children:
                return [child.value for child in self.criteria_tree.root.children]
        else:
            # Получаем дочерние значения для заданного родительского значения
            return self.criteria_tree.get_children_values(parent_value)
        return []

    # === ОСНОВНЫЕ МЕТОДЫ С СОХРАНЕНИЕМ СОСТОЯНИЯ (Без изменений, кроме имени DF) ===

    def add_like(self, object_id):
        if object_id not in self.current_df['ID'].values:
            print(f"❌ Спектакль с ID {object_id} не найден в текущих данных!")
            return False
        if object_id in self.likes:
            print(f"⚠️ Спектакль ID: {object_id} уже в лайках")
            return False
        if object_id in self.dislikes:
            print(f"⚠️ Спектакль ID: {object_id} уже в дизлайках. Сначала удалите дизлайк!")
            return False

        self._save_state(action_type=f"LIKE {object_id}") # Сохраняем ДО изменения для отката
        self.likes.append(object_id)
        print(f"✅ Добавлен лайк спектаклю ID: {object_id}")
        self._update_ranking()
        self._save_state(action_type=f"AFTER_LIKE {object_id}") # Сохраняем ПОСЛЕ изменения
        return True

    def add_dislike(self, object_id):
        if object_id not in self.current_df['ID'].values:
            print(f"❌ Спектакль с ID {object_id} не найден в текущих данных!")
            return False
        if object_id in self.dislikes:
            print(f"⚠️ Спектакль ID: {object_id} уже в дизлайках")
            return False
        if object_id in self.likes:
            print(f"⚠️ Спектакль ID: {object_id} уже в лайках. Сначала удалите лайк!")
            return False

        self._save_state(action_type=f"DISLIKE {object_id}") # Сохраняем ДО изменения для отката
        self.dislikes.append(object_id)
        print(f"❌ Добавлен дизлайк спектаклю ID: {object_id}")
        self._update_ranking()
        self._save_state(action_type=f"AFTER_DISLIKE {object_id}") # Сохраняем ПОСЛЕ изменения
        return True

    def remove_like(self, object_id):
        if object_id in self.likes:
            self._save_state(action_type=f"REMOVE_LIKE {object_id}") # Сохраняем ДО изменения
            self.likes.remove(object_id)
            print(f"🗑️ Удален лайк спектаклю ID: {object_id}")
            self._update_ranking()
            self._save_state(action_type=f"AFTER_REMOVE_LIKE {object_id}") # Сохраняем ПОСЛЕ изменения
            return True
        else:
            print(f"⚠️ Спектакль ID: {object_id} не найден в лайках")
            return False

    def remove_dislike(self, object_id):
        if object_id in self.dislikes:
            self._save_state(action_type=f"REMOVE_DISLIKE {object_id}") # Сохраняем ДО изменения
            self.dislikes.remove(object_id)
            print(f"🗑️ Удален дизлайк спектаклю ID: {object_id}")
            self._update_ranking()
            self._save_state(action_type=f"AFTER_REMOVE_DISLIKE {object_id}") # Сохраняем ПОСЛЕ изменения
            return True
        else:
            print(f"⚠️ Спектакль ID: {object_id} не найден в дизлайках")
            return False

    def apply_filter_df(self, filtered_df, new_filter_criteria):
        """Метод для применения фильтрации с сохранением состояния (для интерактивного режима)."""
        self._save_state(action_type="FILTER")

        self.current_df = filtered_df
        self.filter_criteria = new_filter_criteria # Заменяем, а не обновляем
        self.ranking = None

        # Фильтрация лайков/дизлайков
        current_ids = self.current_df['ID'].values
        self.likes = [like for like in self.likes if like in current_ids]
        self.dislikes = [dislike for dislike in self.dislikes if dislike in current_ids]

        if self.likes or self.dislikes:
            self._update_ranking()
        else:
            print(f"📊 Доступно {len(self.current_df)} спектаклей после фильтрации.")
            self.show_status()

    def reset_system(self):
        """Полный сброс. Не сохраняем в историю, т.к. это отдельная команда сброса."""
        self.current_df = self.initial_df.copy() # Исправлено на initial_df
        self.likes = []
        self.dislikes = []
        self.ranking = None
        self.filter_criteria = {}
        self.action_history = deque(maxlen=20)
        self._save_state()
        print("🔄 Система сброшена к исходному состоянию")
        print(f"📊 Доступно {len(self.initial_df)} спектаклей")

    # --- МЕТОДЫ ДЛЯ ПОКАЗА (без изменений) ---
    def show_spectacles_table(self):
        print("\n🎭 ТЕКУЩИЕ СПЕКТАКЛИ:")
        print("-" * 80)
        display_df = self.current_df[['ID', 'Название постановки', 'Жанр', 'Рейтинг', 'Длительность (мин)', 'Театр']]
        print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.1f'))

    def show_ranking(self, top_n=None):
        if self.ranking is None or len(self.ranking) == 0:
            print("\n❌ РЕЙТИНГ НЕ ДОСТУПЕН")
            if self.likes or self.dislikes:
                 print("⚠️ Нет рекомендаций, соответствующих текущим фильтрам и лайкам/дизлайкам.")
            else:
                 print("⚠️ Добавьте лайки или дизлайки, чтобы получить рейтинг.")
            return

        print("\n🏆 РЕЙТИНГ РЕКОМЕНДАЦИЙ")
        print("=" * 80)
        print(f"Лайки: {self.likes}")
        print(f"Дизлайки: {self.dislikes}")

        # Вывод активных фильтров
        if self.filter_criteria:
            filter_str = ", ".join([f"{k}: {v}" for k, v in self.filter_criteria.items()])
            print(f"Активные фильтры: {filter_str}")

        if top_n is not None and top_n > 0:
            display_ranking = self.ranking.head(top_n)
            print(f"\n📋 ТОП-{len(display_ranking)} РЕКОМЕНДАЦИЙ:")
            print("-" * 120)
        else:
            display_ranking = self.ranking
            print(f"\n📋 ПОЛНЫЙ СПИСОК РЕКОМЕНДАЦИЙ ({len(display_ranking)} объектов):")
            print("-" * 120)

        full_ranking = display_ranking[['Rank', 'ID', 'Название постановки', 'Жанр', 'Стиль', 'Театр', 'Рейтинг', 'Similarity_Score']]
        print(tabulate(full_ranking, headers='keys', tablefmt='grid', floatfmt='.3f'))

    # ИСПРАВЛЕНИЕ: show_status с правильным отступом
    def show_status(self):
        print("\n📊 ТЕКУЩЕЕ СОСТОЯНИЕ СИСТЕМЫ")
        print("=" * 40)
        print(f"- Исходное количество спектаклей: {len(self.initial_df)}")
        print(f"- Текущее количество спектаклей: {len(self.current_df)}")
        print(f"- Лайки: {self.likes}")
        print(f"- Дизлайки: {self.dislikes}")
        print(f"- Активные фильтры: {self.filter_criteria}")
        print(f"- Есть рейтинг: {'Да' if self.ranking is not None else 'Нет'}")
        if self.ranking is not None:
             print(f"- Количество рекомендаций: {len(self.ranking)}")

    # === МЕТОДЫ ДЛЯ АГРЕГАЦИИ И ОБЩИХ ВОПРОСОВ ===

    def get_aggregation_result(self, query, filters=None):
        """Обрабатывает запросы на агрегацию (сравнение)."""
        query_lower = query.lower()
        df = self.current_df.copy()

        # Применяем фильтры, если они есть
        if filters:
            for filter_item in filters:
                col = filter_item['column']
                op = filter_item['operator']
                val = filter_item['value']

                if col in ['Рейтинг', 'Длительность (мин)']:
                    try:
                        df = df.query(f"`{col}` {op} {val}")
                    except:
                        pass
                else:
                    if op == '==':
                        df = df[df[col] == val]

        if df.empty:
            print("❌ Не найдено записей, соответствующих критериям.")
            return None

        # Определяем тип агрегации
        if 'самый высокий' in query_lower or 'максимальный' in query_lower:
            if 'рейтинг' in query_lower:
                max_rating = df['Рейтинг'].max()
                result_df = df[df['Рейтинг'] == max_rating]
                print(f"\n🏆 СПЕКТАКЛИ С САМЫМ ВЫСОКИМ РЕЙТИНГОМ ({max_rating}):")
            elif 'театр' in query_lower:
                # Группируем по театрам и находим театр с максимальным средним рейтингом
                theatre_ratings = df.groupby('Театр')['Рейтинг'].mean()
                max_theatre = theatre_ratings.idxmax()
                max_rating = theatre_ratings.max()
                result_df = df[df['Театр'] == max_theatre]
                print(f"\n🏆 ТЕАТР С САМЫМ ВЫСОКИМ СРЕДНИМ РЕЙТИНГОМ: {max_theatre} (средний рейтинг: {max_rating:.2f})")
            else:
                print("⚠️ Не удалось определить тип агрегации.")
                return None
        elif 'самый низкий' in query_lower or 'минимальный' in query_lower:
            if 'рейтинг' in query_lower:
                min_rating = df['Рейтинг'].min()
                result_df = df[df['Рейтинг'] == min_rating]
                print(f"\n📉 СПЕКТАКЛИ С САМЫМ НИЗКИМ РЕЙТИНГОМ ({min_rating}):")
            else:
                print("⚠️ Не удалось определить тип агрегации.")
                return None
        else:
            print("⚠️ Не удалось определить тип агрегации.")
            return None

        display_df = result_df[['ID', 'Название постановки', 'Жанр', 'Рейтинг', 'Театр']]
        print(tabulate(display_df, headers='keys', tablefmt='grid', floatfmt='.1f'))
        return result_df

    def answer_general_question(self, query):
        """Отвечает на общие вопросы о системе."""
        query_lower = query.lower()

        if 'как мне начать' in query_lower or 'начать пользоваться' in query_lower:
            print("\n📖 КАК НАЧАТЬ ПОЛЬЗОВАТЬСЯ СИСТЕМОЙ:")
            print("1. Добавьте лайки к понравившимся спектаклям (например: 'Поставь лайк Кармен')")
            print("2. Добавьте дизлайки к непонравившимся спектаклям")
            print("3. Система автоматически создаст рейтинг рекомендаций на основе ваших предпочтений")
            print("4. Используйте фильтры для поиска спектаклей по критериям")
            print("5. Используйте 'шаг назад' для отката последнего действия")

        elif 'объясните' in query_lower and ('лайк' in query_lower or 'дизлайк' in query_lower):
            print("\n💡 О ЛАЙКАХ И ДИЗЛАЙКАХ:")
            print("- Вы можете ставить лайки и дизлайки всем постановкам в базе данных")
            print("- Лайки увеличивают схожесть с похожими спектаклями")
            print("- Дизлайки уменьшают схожесть с похожими спектаклями")
            print("- Система использует многомерные метрики схожести для расчета рекомендаций")
            print("- Вы можете удалить лайк/дизлайк командой 'Сними лайк/дизлайк'")

        elif 'основные правила' in query_lower or 'как работают' in query_lower:
            print("\n📋 ОСНОВНЫЕ ПРАВИЛА СИСТЕМЫ:")
            print("1. ЛАЙКИ: Увеличивают схожесть с похожими спектаклями")
            print("2. ДИЗЛАЙКИ: Уменьшают схожесть с похожими спектаклями (штраф 0.5x)")
            print("3. ФИЛЬТРЫ: Позволяют ограничить набор спектаклей для поиска")
            print("4. ОТКАТ: Команда 'шаг назад' отменяет последнее действие")
            print("5. СБРОС: Команда 'сбрось все' возвращает систему в начальное состояние")
            print("6. РЕЙТИНГ: Автоматически обновляется при изменении лайков/дизлайков")

        elif 'что произойдет' in query_lower and 'лайкнуть' in query_lower:
            print("\n⚠️ ПОВТОРНЫЙ ЛАЙК:")
            print("Если вы попытаетесь лайкнуть уже понравившийся спектакль, система сообщит:")
            print("'⚠️ Спектакль ID: X уже в лайках'")
            print("Аналогично для дизлайков.")

        elif 'вес' in query_lower or 'важность' in query_lower or 'признак' in query_lower:
            print("\n⚖️ ВЕСА ПРИЗНАКОВ В РАСЧЕТЕ СХОЖЕСТИ:")
            max_weight = max(self.FEATURE_WEIGHTS.values())
            max_feature = [k for k, v in self.FEATURE_WEIGHTS.items() if v == max_weight][0]
            print(f"Самый высокий вес: {max_feature} = {max_weight}")
            print("\nВсе веса признаков:")
            for feature, weight in sorted(self.FEATURE_WEIGHTS.items(), key=lambda x: x[1], reverse=True):
                print(f"  - {feature}: {weight}")
            print(f"  - Бинарные признаки (всего {len(self.BINARY_COLUMNS)}): 0.05 (распределено равномерно)")
        else:
            print("❓ Прошу прощения, я не могу ответить на ваш вопрос.")
            return False

        return True

    def process_like_dislike_and_rank(self, query, current_df, entity_extractor):
        """Обрабатывает комбинированные запросы: лайки/дизлайки + рекомендация."""
        # Сохраняем состояние перед обработкой комбинированного запроса
        self._save_state(action_type="LIKE_DISLIKE_AND_RANK")

        # Извлекаем все ID из запроса (и по числам, и по названиям)
        ids_found = []
        query_lower = query.lower()

        # Ищем числовые ID
        id_matches = re.findall(r'\b(\d+)\b', query)
        for id_str in id_matches:
            try:
                id_val = int(id_str)
                if id_val in current_df['ID'].values and id_val not in ids_found:
                    ids_found.append(id_val)
            except:
                pass

        # Ищем по названиям спектаклей (используем метод extract_multiple_ids, который поддерживает fuzzy matching)
        ids_by_title = entity_extractor.extract_multiple_ids(query, current_df)
        for id_val in ids_by_title:
            if id_val not in ids_found:
                ids_found.append(id_val)

        if not ids_found:
            print("❌ Не удалось найти спектакли в запросе.")
            return

        # Определяем лайки и дизлайки
        likes_to_add = []
        dislikes_to_add = []

        # Разбиваем запрос на части по запятым и точкам
        # Используем более точное разбиение, чтобы сохранить контекст
        parts = []
        current_part = ""
        for char in query_lower:
            if char in [',', '.']:
                if current_part.strip():
                    parts.append(current_part.strip())
                current_part = ""
            else:
                current_part += char
        if current_part.strip():
            parts.append(current_part.strip())

        # Если не удалось разбить, используем исходный запрос
        if not parts:
            parts = [query_lower]

        for part in parts:
            part = part.strip()
            if not part:
                continue

            # Определяем тип действия для этой части
            is_dislike_part = 'дизлайк' in part or 'не нравится' in part or 'не_нравится' in part
            is_like_part = 'лайк' in part or ('нравится' in part and 'не' not in part)

            # Ищем ID в этой части
            for id_val in ids_found:
                id_str = str(id_val)
                # Проверяем, упоминается ли ID в этой части
                if id_str in part:
                    if is_dislike_part:
                        if id_val not in dislikes_to_add:
                            dislikes_to_add.append(id_val)
                    elif is_like_part:
                        if id_val not in likes_to_add:
                            likes_to_add.append(id_val)
                else:
                    # Проверяем по названию спектакля
                    try:
                        title = str(current_df[current_df['ID'] == id_val]['Название постановки'].iloc[0]).lower()
                        if title in part or any(word in part for word in title.split() if len(word) > 3):
                            if is_dislike_part:
                                if id_val not in dislikes_to_add:
                                    dislikes_to_add.append(id_val)
                            elif is_like_part:
                                if id_val not in likes_to_add:
                                    likes_to_add.append(id_val)
                    except:
                        pass

        # Если не удалось определить по частям, используем общий контекст вокруг каждого ID
        if not likes_to_add and not dislikes_to_add:
            for id_val in ids_found:
                id_str = str(id_val)
                id_pos = query_lower.find(id_str)
                if id_pos == -1:
                    # Ищем по названию
                    title = str(current_df[current_df['ID'] == id_val]['Название постановки'].iloc[0]).lower()
                    id_pos = query_lower.find(title)

                if id_pos >= 0:
                    context = query_lower[max(0, id_pos-40):id_pos+40]
                    if 'дизлайк' in context or 'не нравится' in context or 'не_нравится' in context:
                        dislikes_to_add.append(id_val)
                    else:
                        likes_to_add.append(id_val)
                else:
                    # По умолчанию - лайк
                    likes_to_add.append(id_val)

        # Применяем лайки и дизлайки (без сохранения состояния, т.к. уже сохранили выше)
        print(f"\n📝 Обработка комбинированного запроса...")
        for like_id in likes_to_add:
            if like_id in current_df['ID'].values:
                # Временно отключаем сохранение состояния в add_like
                was_in_likes = like_id in self.likes
                if not was_in_likes and like_id not in self.dislikes:
                    self.likes.append(like_id)
                    print(f"✅ Добавлен лайк спектаклю ID: {like_id}")

        for dislike_id in dislikes_to_add:
            if dislike_id in current_df['ID'].values:
                # Временно отключаем сохранение состояния в add_dislike
                was_in_dislikes = dislike_id in self.dislikes
                if not was_in_dislikes and dislike_id not in self.likes:
                    self.dislikes.append(dislike_id)
                    print(f"❌ Добавлен дизлайк спектаклю ID: {dislike_id}")

        # Сохраняем состояние после всех изменений
        self._save_state(action_type="AFTER_LIKE_DISLIKE_AND_RANK")

        # Обновляем рейтинг без показа (покажем только топ-5)
        if self.likes or self.dislikes:
            self._update_ranking(show_ranking=False)
            print("✅ Рейтинг обновлен!")
            print(f"💡 Создано {len(self.ranking)} рекомендаций")
            # Показываем только топ-5
            if self.ranking is not None and len(self.ranking) > 0:
                print(f"\n💡 РЕКОМЕНДАЦИИ НА ОСНОВЕ ВАШИХ ПРЕДПОЧТЕНИЙ:")
                self.show_ranking(top_n=5)
            else:
                print("⚠️ Не удалось создать рекомендации. Добавьте больше лайков/дизлайков.")
        else:
            print("⚠️ Не удалось создать рекомендации. Добавьте больше лайков/дизлайков.")

    def remove_filter(self):
        """Снимает все активные фильтры."""
        self._save_state(action_type="REMOVE_FILTER")
        self.current_df = self.initial_df.copy()
        self.filter_criteria = {}
        self.ranking = None
        print("✅ Все фильтры сняты. Возвращены все спектакли.")
        print(f"📊 Доступно {len(self.current_df)} спектаклей")

        if self.likes or self.dislikes:
            self._update_ranking()
        else:
            self.show_status()