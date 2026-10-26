"""
Морфологический и синтаксический анализ для NLP.
"""

from config import PYMORPHY2_AVAILABLE, SPACY_AVAILABLE, nlp_spacy, pymorphy2


class MorphologicalAnalyzer:
    """Морфологический анализ с помощью pymorphy2."""

    def __init__(self):
        if PYMORPHY2_AVAILABLE:
            try:
                self.morph = pymorphy2.MorphAnalyzer()
            except (AttributeError, ImportError) as e:
                # Обработка ошибок совместимости (например, Python 3.11+ с старым pymorphy2)
                print(f"⚠️ Не удалось инициализировать pymorphy2: {e}")
                print("💡 Попробуйте обновить pymorphy2: pip install --upgrade pymorphy2")
                self.morph = None
        else:
            self.morph = None

    def normalize(self, word):
        """Приводит слово к нормальной форме (лемматизация)."""
        if not self.morph:
            return word.lower()

        try:
            parsed = self.morph.parse(word)[0]
            return parsed.normal_form
        except:
            return word.lower()

    def get_grammar_info(self, word):
        """Получает грамматическую информацию о слове."""
        if not self.morph:
            return {}

        try:
            parsed = self.morph.parse(word)[0]
            return {
                'normal_form': parsed.normal_form,
                'tag': str(parsed.tag),
                'POS': parsed.tag.POS,  # Часть речи
                'case': parsed.tag.case,  # Падеж
                'number': parsed.tag.number,  # Число
            }
        except:
            return {}

    def normalize_text(self, text):
        """Нормализует весь текст."""
        if not self.morph:
            return text.lower()

        words = text.split()
        normalized = [self.normalize(word) for word in words]
        return ' '.join(normalized)


class SyntacticAnalyzer:
    """Синтаксический анализ с помощью spaCy."""

    def __init__(self):
        self.nlp = nlp_spacy if SPACY_AVAILABLE and nlp_spacy else None

    def analyze(self, text):
        """Выполняет синтаксический анализ текста."""
        if not self.nlp:
            return None

        try:
            doc = self.nlp(text)
            return doc
        except Exception as e:
            print(f"⚠️ Ошибка синтаксического анализа: {e}")
            return None

    def extract_dependencies(self, text):
        """Извлекает синтаксические зависимости."""
        doc = self.analyze(text)
        if not doc:
            return []

        dependencies = []
        for token in doc:
            dependencies.append({
                'text': token.text,
                'lemma': token.lemma_,
                'pos': token.pos_,
                'dep': token.dep_,
                'head': token.head.text,
                'head_pos': token.head.pos_,
            })
        return dependencies

    def find_verb_objects(self, text):
        """Находит объекты глаголов (для поиска действий)."""
        doc = self.analyze(text)
        if not doc:
            return []

        objects = []
        for token in doc:
            if token.pos_ == 'VERB':
                # Ищем прямые дополнения (obj) и косвенные (obl)
                for child in token.children:
                    if child.dep_ in ['obj', 'obl', 'nsubj']:
                        objects.append({
                            'verb': token.lemma_,
                            'object': child.text,
                            'object_lemma': child.lemma_,
                            'relation': child.dep_,
                        })
        return objects

    def find_noun_phrases(self, text):
        """Находит именные группы (для поиска сущностей)."""
        doc = self.analyze(text)
        if not doc:
            return []

        phrases = []
        try:
            # Пытаемся использовать noun_chunks (может быть недоступно для русской модели)
            for chunk in doc.noun_chunks:
                phrases.append({
                    'text': chunk.text,
                    'root': chunk.root.text,
                    'root_lemma': chunk.root.lemma_,
                    'root_pos': chunk.root.pos_,
                })
        except (AttributeError, ValueError, TypeError, Exception):
            # Если noun_chunks недоступен (например, для русской модели spaCy),
            # извлекаем существительные вручную
            # Ищем существительные (NOUN) и их зависимые слова
            for token in doc:
                if token.pos_ == 'NOUN' or token.pos_ == 'PROPN':  # Существительные и имена собственные
                    # Собираем фразу из существительного и его зависимых
                    phrase_words = [token]
                    for child in token.children:
                        if child.dep_ in ['amod', 'det', 'nummod']:  # Прилагательные, определители, числительные
                            phrase_words.append(child)
                    
                    # Сортируем по позиции в предложении
                    phrase_words.sort(key=lambda x: x.i)
                    phrase_text = ' '.join([w.text for w in phrase_words])
                    
                    phrases.append({
                        'text': phrase_text,
                        'root': token.text,
                        'root_lemma': token.lemma_,
                        'root_pos': token.pos_,
                    })
        return phrases

