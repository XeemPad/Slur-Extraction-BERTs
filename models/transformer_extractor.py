from typing import List, Set, Optional, Tuple, Dict, Any
import re
from pathlib import Path
from tqdm import tqdm

import numpy as np
import pandas as pd

from transformers import AutoTokenizer, AutoModelForTokenClassification, Trainer, TrainingArguments
from torch.utils.data import Dataset
from peft import get_peft_model, LoraConfig, TaskType
import torch

from models.base_extractor import BaseSlurExtractor


class TransformerSlurExtractor(BaseSlurExtractor):
    def __init__(
        self,
        output_dir: Path,
        train_args: dict,
        model_name: str=None,
        max_length: int = 128,
        **kwargs
    ):
        super().__init__(**kwargs)
        self.model_name = model_name if model_name else self.__class__.__name__
        self.max_length = max_length
        self.train_args: dict = train_args
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.output_dir = output_dir
        
        # Инициализация модели
        self._setup_model()

    def fit(self, texts: List[str], spans_list: List[List[Tuple[int, int]]]):
        """Обучение модели - принимает готовый датасет"""
        if self.verbose:
            print("=== Обучение Transformer модели ===")
            print(f"Размер датасета: {len(texts)}")

        # Подготовка датасета

        dataset = SlurTokenClassificationDataset(
            texts, spans_list, self.tokenizer, max_length=self.max_length,
            gusev_rubert=self.model_name == "IlyaGusev/rubertconv_toxic_editor"
        )

        # Обучение
        self._train_model(dataset)
        
        self.is_fitted = True
        return self

    def transform(self, texts: List[str]) -> List[str]:
        if not self.is_fitted:
            raise ValueError("Model is not fitted yet!")

        self.model.eval()
        results = []
        for text in tqdm(texts, desc="Extracting slurs"):
            enc = self._tokenize(text)
            preds = self._predict(enc)
            raw_spans = self._get_spans(enc['offset_mapping'][0], preds)
            merged = self._merge_spans(raw_spans)
            slurs = self._extract_slurs(text, merged)
            results.append(','.join(sorted(slurs)) if slurs else '')
        return results
    
    def _setup_model(self):
        """Настройка модели и токенизатора"""
        if self.verbose:
            print(f"Устройство: {self.device}")
            print(f"Загружаем модель: {self.model_name}")
        
        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token
        
        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)
        # заморозить encoder.layer.0 … encoder.layer.7
        for name, param in self.model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8)):
                param.requires_grad = False

        if self.verbose:
            print("Модель загружена")
    
    def _train_model(self, dataset):
        """Запуск обучения"""
        training_args = self._get_training_args()
        self.trainer = self._create_trainer(dataset, training_args)

        if self.verbose:
            print("Начинаем обучение...")

        self.trainer.train()

        if self.verbose:
            print("Обучение завершено!")
    
    def _get_training_args(self):
        """Настройки обучения"""
        epochs = self.train_args.get('epochs', 3)
        batch_size = self.train_args.get('batch_size', 16)
        learning_rate = self.train_args.get('learning_rate', 1e-5)
        warmup_steps = self.train_args.get('warmup_steps', 200)
        weight_decay = self.train_args.get('weight_decay', 0.01)
        logging_steps = self.train_args.get('logging_steps', 50)
        grad_accum_steps = self.train_args.get('grad_accum_steps', 2)

        
        return TrainingArguments(
            output_dir=self.output_dir / f"{self.__class__.__name__}",
            
            num_train_epochs=epochs,
            per_device_train_batch_size=batch_size,
            
            learning_rate=learning_rate,
            warmup_steps=warmup_steps,
            weight_decay=weight_decay,
            gradient_accumulation_steps=grad_accum_steps,

            logging_steps=logging_steps,
            save_strategy="no",
            report_to="none",
            fp16=False,
        )
    
    def _create_trainer(self, dataset, training_args):
        """Создает trainer"""
        return WeightedTrainer(
            model=self.model,
            args=training_args,
            train_dataset=dataset,
            processing_class=self.tokenizer
        )


    def _tokenize(self, text: str) -> Dict[str, Any]:
        enc = self.tokenizer(
            text,
            truncation=True,
            padding='max_length',
            max_length=self.max_length,
            return_offsets_mapping=True,
            return_tensors='pt'
        ).to(self.device)
        return enc

    def _predict(self, enc: Dict[str, Any]) -> List[int]:
        inputs = {k: v for k, v in enc.items() if k != 'offset_mapping'}
        with torch.no_grad():
            logits = self.model(**inputs).logits
        preds = torch.argmax(logits, dim=-1)[0].tolist()
        return preds

    def _get_spans(self, offsets: torch.Tensor, preds: List[int]) -> List[Tuple[int, int]]:
        spans = []
        for (start, end), p in zip(offsets.tolist(), preds):
            if p == 1 and start < end:
                spans.append((start, end))
        return spans

    def _merge_spans(self, spans: List[Tuple[int, int]]) -> List[Tuple[int, int]]:
        if not spans:
            return []
        spans = sorted(spans)
        merged = []
        curr_s, curr_e = spans[0]
        for s, e in spans[1:]:
            if s <= curr_e + 1:
                curr_e = max(curr_e, e)
            else:
                merged.append((curr_s, curr_e))
                curr_s, curr_e = s, e
        merged.append((curr_s, curr_e))
        return merged

    def _extract_slurs(self, text: str, spans: List[Tuple[int, int]]) -> List[str]:
        slurs = {text[s:e].lower().strip() for s, e in spans if e - s >= 2}
        return list(slurs)


class WeightedTrainer(Trainer):
    """Trainer с взвешенной функцией потерь по классам"""
    step = 0  # Для kaggle где в контейнере не работает progress-bar

    def compute_loss(self, model, inputs, return_outputs=False, num_items_in_batch=None):
        labels = inputs.get('labels')
        # forward pass
        outputs = model(**{k: v for k, v in inputs.items() if k != 'labels'})

        if hasattr(outputs, 'logits'):
            # обычная токен-классификация
            logits = outputs.logits

            mask = labels != -100
            pos = (labels == 1) & mask
            neg = (labels == 0) & mask
            pos_weight = neg.sum().float() / (pos.sum().float() + 1e-6)

            label_weights = [1.0, pos_weight]
            loss_fct = torch.nn.CrossEntropyLoss(
                weight=torch.tensor(label_weights, device=logits.device),
                ignore_index=-100,
            )
            loss = loss_fct(logits.view(-1, 2), labels.view(-1))
        else:
            # CRF возвращает сразу loss
            loss = outputs

        self.step += 1
        if self.step % 50 == 0:
            print(f"Step {self.step}. Train loss: {loss}")
        return (loss, outputs) if return_outputs else loss


class LoRASlurExtractor(TransformerSlurExtractor):
    def _setup_model(self):
        """Настройка модели и токенизатора"""
        if self.verbose:
            print(f"Устройство: {self.device}")
            print(f"Загружаем модель: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)

        # заморозить encoder.layer.0 … encoder.layer.7
        for name, param in self.model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8)):
                param.requires_grad = False

        # Конфигурируем LoRA
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8, # ранг low-rank
            lora_alpha=32, # шкалирование
            lora_dropout=0.1,
            target_modules=["query", "value"]  # какие подмодули адаптировать
        )

        self.model = get_peft_model(self.model, peft_config)

        if self.verbose:
            print("Модель загружена")

# Конкретные модели
class RuBertSlurExtractor(TransformerSlurExtractor):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="DeepPavlov/rubert-base-cased",
            **kwargs
        )

class RoSBertaSlurExtractor(TransformerSlurExtractor):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="ai-forever/ru-en-RoSBERTa",
            **kwargs
        )

class RoSBertaLoRASlurExtractor(LoRASlurExtractor):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="ai-forever/ru-en-RoSBERTa",
            **kwargs
        )

class RuBertLoRASlurExtractor(LoRASlurExtractor):
    def __init__(self, **kwargs):
        super().__init__(
            model_name="IlyaGusev/rubertconv_toxic_editor",
            **kwargs
        )

    def _setup_model(self):
        """Настройка модели и токенизатора"""
        if self.verbose:
            print(f"Устройство: {self.device}")
            print(f"Загружаем модель: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=4,
            ignore_mismatched_sizes=False
        ).to(self.device)

        # заморозить encoder.layer.0 -- encoder.layer.7
        for name, param in self.model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8)):
                param.requires_grad = False

        # Конфигурируем LoRA
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8, # ранг low-rank
            lora_alpha=32, # шкалирование
            lora_dropout=0.1,
            target_modules=["query", "value"]  # какие подмодули адаптировать
        )

        self.model = get_peft_model(self.model, peft_config)

        if self.verbose:
            print("Модель загружена")


class FineTuneRoSBERTa(RoSBertaLoRASlurExtractor):
    def __init__(self, *args, model_saved_dict=None, **kwargs):
        self.model_saved_dict = model_saved_dict
        super().__init__(*args, **kwargs)

    def _setup_model(self):
        """Настройка модели и токенизатора"""
        if self.verbose:
            print(f"Устройство: {self.device}")
            print(f"Загружаем модель: {self.model_name}")

        self.tokenizer = AutoTokenizer.from_pretrained(self.model_name)
        if self.tokenizer.pad_token is None:
            self.tokenizer.pad_token = self.tokenizer.eos_token

        self.model = AutoModelForTokenClassification.from_pretrained(
            self.model_name,
            num_labels=2,
            ignore_mismatched_sizes=True
        ).to(self.device)

        # заморозить encoder.layer.0 … encoder.layer.7 (иначе веса не подгружаются)
        for name, param in self.model.named_parameters():
            if any(f"encoder.layer.{i}" in name for i in range(8)):
                param.requires_grad = False

        # Конфигурируем LoRA
        peft_config = LoraConfig(
            task_type=TaskType.TOKEN_CLS,
            inference_mode=False,
            r=8, # ранг low-rank
            lora_alpha=32, # шкалирование
            lora_dropout=0.1,
            target_modules=["query", "value"]  # какие подмодули адаптировать
        )

        self.model = get_peft_model(self.model, peft_config)

        # Восстановим веса модели
        if self.model_saved_dict is not None:
            model_state = self.model_saved_dict.get('model_state_dict')
            if model_state is not None:
                self.model.load_state_dict(model_state)

        if self.verbose:
            print("Модель загружена")

    def _train_model(self, dataset):
        """Запуск обучения"""
        training_args = self._get_training_args()
        self.trainer = self._create_trainer(dataset, training_args)

        if self.verbose:
            print("Начинаем обучение...")

        self.trainer.train()

        if self.verbose:
            print("Обучение завершено!")
