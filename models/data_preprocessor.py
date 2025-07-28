import pandas as pd
from typing import List, Tuple, Set, Dict, Any, Optional
from tqdm import tqdm

from transformers import AutoTokenizer
import torch
from torch.utils.data import Dataset
from sklearn.base import BaseEstimator, TransformerMixin


class SlurDataPreprocessor:
    """Класс для предобработки данных, без привязки к токенизатору"""
    
    def __init__(self, max_length: int=128, verbose: bool=True):
        self.max_length = max_length
        self.verbose = verbose
    
    def prepare_training_data(self, texts: List[str], slur_labels: Optional[List[str]],
                              tokenizer: AutoTokenizer) -> List[Dict[str, Any]]:
        """Готовит данные — slur_labels может быть None для инференса"""
        training_data = []
        too_long_count = 0

        zip_obj = zip(texts, slur_labels or ['']*len(texts))
        iterator = tqdm(zip_obj, desc="Preparing data") if self.verbose else zip_obj

        for text, slur_str in iterator:
            slurs = self._parse_slur_labels(slur_str)

            # собираем все spans по каждому слову-мату
            spans: List[Tuple[int,int]] = []
            for slur in slurs:
                for m in re.finditer(re.escape(slur), text, flags=re.IGNORECASE):
                    spans.append((m.start(), m.end()))

            # если текст слишком длинный - обрезаем
            if self._is_text_too_long(text, tokenizer):
                text = self._truncate_text(text)

            token_data = self._create_token_labels(text, slurs, tokenizer)
            training_data.append({
                'text': text,
                'spans': spans
            })

        if self.verbose and too_long_count > 0:
            print(f"!!! Обрезано {too_long_count} длинных текстов из {len(texts)}")

        return training_data
    
    def _is_text_too_long(self, text: str, tokenizer: AutoTokenizer) -> bool:
        """Проверяет длину с переданным токенизатором"""
        quick_tokens = tokenizer.tokenize(text)
        return len(quick_tokens) > self.max_length - 2

    '''
    def _create_token_labels(self, text: str, slurs: Set[str], tokenizer: AutoTokenizer) -> Dict[str, List]:
        """Создает метки с переданным токенизатором"""
        encoding = tokenizer(
            text,
            truncation=True,
            max_length=self.max_length - 2,
            add_special_tokens=False,
            return_tensors=None
        )
        
        tokens = tokenizer.convert_ids_to_tokens(encoding['input_ids'])
        labels = []
        
        for token in tokens:
            # Универсальная очистка для разных токенизаторов
            clean_token = self._clean_token(token).lower()
            is_slur = any(clean_token in slur or slur in clean_token for slur in slurs)
            labels.append(1 if is_slur else 0)
        
        return {'tokens': tokens, 'labels': labels}
    '''

    def _clean_token(self, token: str) -> str:
        """Универсальная очистка токена"""
        # BERT subwords
        token = token.replace('##', '')
        # RoBERTa subwords  
        token = token.replace('Ġ', '')
        # Другие префиксы/суффиксы
        return token.strip()

    def _parse_slur_labels(self, slur_str: str) -> Set[str]:
        """Парсит строку с метками матов"""
        if pd.isna(slur_str) or slur_str == '':
            return set()
        return set(s.strip().lower() for s in str(slur_str).split(',') if s.strip())

    def _truncate_text(self, text: str) -> str:
        """Обрезает слишком длинный текст"""
        return text[:self.max_length * 4]  # ~4 символа на токен


class SlurDataset(Dataset): # Unused
    """Dataset для обучения моделей извлечения матов"""
    
    def __init__(self, training_data: List[Dict], tokenizer: AutoTokenizer, max_length: int = 128):
        self.data = training_data
        self.tokenizer = tokenizer
        self.max_length = max_length
    
    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        item = self.data[idx]
        
        # Токенизация текста
        encoding = self.tokenizer(
            item['text'],
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_tensors='pt'
        )
        
        # Подготовка меток
        labels = self._prepare_labels(item['labels'])
        
        return {
            'input_ids': encoding['input_ids'].flatten(),
            'attention_mask': encoding['attention_mask'].flatten(),
            'labels': torch.tensor(labels, dtype=torch.long)
        }
    
    def _prepare_labels(self, token_labels: List[int]) -> List[int]:
        """Подготавливает метки под формат модели"""
        # Обрезаем метки под размер (с учетом [CLS] и [SEP])
        labels = token_labels[:self.max_length - 2]
        
        # Добавляем метки для специальных токенов
        final_labels = [0]  # [CLS]
        final_labels.extend(labels)  # Основные токены
        
        # Дополняем до max_length
        while len(final_labels) < self.max_length:
            final_labels.append(0)  # [SEP] и [PAD]
        
        return final_labels[:self.max_length]



class SlurDataTransformer(BaseEstimator, TransformerMixin):
    """Sklearn-compatible трансформер для подготовки данных"""
    
    def __init__(self, tokenizer, max_length=128, verbose=True):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.verbose = verbose
        self.preprocessor = SlurDataPreprocessor(max_length, verbose)
    
    def fit(self, X, y=None):
        """Ничего не делаем при fit"""
        return self
    
    def transform(self, X, y=None):
        """Трансформируем тексты в training_data"""
        if hasattr(X, 'values'):  # pandas Series
            texts = X.values.tolist()
        elif isinstance(X, list):
            texts = X
        else:
            texts = list(X)
        
        return self.preprocessor.prepare_training_data(texts, None, self.tokenizer)



class SlurTokenClassificationDataset(Dataset):
    """
    Dataset для токен-классификации: берет тексты и спаны матов,
    возвращает входы для модели и метки по offset_mapping.
    """
    def __init__(
            self,
            texts: List[str],
            spans_list: List[List[Tuple[int, int]]],
            tokenizer: AutoTokenizer,
            max_length: int = 128,
            gusev_rubert: bool = False
        ):
        self.texts = texts
        self.spans_list = spans_list
        self.tokenizer = tokenizer
        self.max_length = max_length
        self.gusev_rubert = gusev_rubert

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        spans = self.spans_list[idx]  # Список (start, end) мата в тексте

        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt',
        )
        offsets = enc.pop('offset_mapping')[0]
        labels = torch.zeros(len(offsets), dtype=torch.long)

        # Помечаем все токены, пересекающиеся со span'ами матов
        for (m_start, m_end) in spans:
            for i, (t_start, t_end) in enumerate(offsets.tolist()):
                if t_start < m_end and t_end > m_start:
                    labels[i] = 1 if not self.gusev_rubert else 2  # mark as Delete if gusev_rubert is used

        # Спецтокены и паддинг -> -100
        special_ids = set(self.tokenizer.all_special_ids)
        for i, token_id in enumerate(enc['input_ids'][0].tolist()):
            if token_id in special_ids:
                labels[i] = -100

        # Формируем батч
        item = {k: v.squeeze(0) for k, v in enc.items()}
        item['labels'] = labels
        return item


class SlurDatasetCreator(BaseEstimator, TransformerMixin): # Unused
    """Создает PyTorch Dataset из preprocessed данных"""
    
    def __init__(self, tokenizer, max_length=128):
        self.tokenizer = tokenizer
        self.max_length = max_length
        self._mode = 'train'  # train или predict
    
    def fit(self, X, y=None):
        self._mode = 'train'
        return self
    
    def transform(self, X, y=None):
        """X - это training_data из предыдущего шага"""
        if self._mode == 'train':
            # При обучении создаем Dataset
            return SlurDataset(X, self.tokenizer, self.max_length)
        else:
            # При predict просто передаем данные дальше
            return X
    
    def predict(self, X):
        """Для predict режима"""
        self._mode = 'predict'
        return self.transform(X)


class SklearnCompatibleExtractor(BaseEstimator):
    """Sklearn-compatible обертка для трансформерных экстракторов"""
    
    def __init__(self, extractor_class, output_dir, **extractor_kwargs):
        self.extractor_class = extractor_class
        self.output_dir = output_dir
        self.extractor_kwargs = extractor_kwargs
        self.extractor = None
    
    def fit(self, X, y=None):
        """X - это dataset"""
        self.extractor = self.extractor_class(
            output_dir=self.output_dir,
            **self.extractor_kwargs
        )
        self.extractor.fit(X)  # X уже SlurDataset
        return self
    
    def predict(self, X):
        """X - это training_data от preprocessor"""
        if self.extractor is None:
            raise ValueError("Модель не обучена!")
        
        # Извлекаем тексты из training_data
        texts = [item['text'] for item in X]
        return self.extractor.transform(texts)
    
    def score(self, X, y):
        """Вычисляет метрику"""
        predictions = self.predict(X)
        return self.extractor.evaluate_predictions(predictions, y)