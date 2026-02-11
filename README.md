# Pneumonia Detection

## Постановка задачи

Разработка системы автоматической диагностики пневмонии на основе рентгеновских снимков грудной клетки с использованием методов компьютерного зрения и машинного обучения. Система предназначена для помощи медицинским специалистам в быстрой и точной диагностике пневмонии, что критически важно для своевременного начала лечения и снижения смертности от данного заболевания.

## Формат входных и выходных данных

**Входные данные:**
- Рентгеновские снимки грудной клетки в формате JPEG
- Размер изображений: переменный (автоматически приводится к 224x224 пикселей)
- Цветовое пространство: RGB

**Выходные данные:**
- Класс предсказания: `NORMAL` (здоров) или `PNEUMONIA` (пневмония)
- Уровень уверенности модели: значение от 0.0 до 1.0

**Протокол взаимодействия:**
- REST API через Triton Inference Server
- CLI интерфейс для локального инференса

## Метрики

**Основные метрики:**
- **Accuracy**: ≥ 90%
- **Precision**: ≥ 88%
- **Recall**: ≥ 92%
- **F1-Score**: ≥ 90%

**Обоснование:** Высокий recall критичен в медицинской диагностике для минимизации пропущенных случаев пневмонии. Precision также важен для избежания ненужных медицинских вмешательств.

## Валидация и тест

**Стратегия разделения:**
- **Train**: 70% данных (стратифицированное разделение)
- **Validation**: 15% данных (для подбора гиперпараметров)
- **Test**: 15% данных (финальная оценка модели)

**Методы валидации:**
- Стратифицированное разделение для сохранения пропорций классов
- Cross-validation не используется из-за специфики медицинских данных
- Фиксированный random_seed=42 для воспроизводимости

**Воспроизводимость:**
- Все случайные состояния зафиксированы
- Версии библиотек закреплены в pyproject.toml
- Конфигурации сохраняются в Hydra configs
- MLflow для трекинга экспериментов

## Датасеты

**Основной датасет:** [Chest X-Ray Images (Pneumonia)](https://www.kaggle.com/datasets/paultimothymooney/chest-xray-pneumonia)

**Характеристики:**
- **Объем**: 5,856 изображения
- **Размер**: 1.24 GB
- **Классы**: NORMAL, PNEUMONIA

**Особенности данных:**
- Дисбаланс классов (соотношение ~1:2.7)
- Различное качество и разрешение снимков

**Потенциальные проблемы:**
- Необходимость балансировки классов (используется RandomOverSampler)
- Различия в оборудовании и условиях съемки

## Моделирование

### Бейзлайн

Логистическая регрессия на признаках, извлеченных с помощью предобученной ResNet-18.

### Основная модель

**Архитектура:** Vision Transformer (ViT) - `google/vit-base-patch16-224`

**Технические детали:**
- **Предобучение**: ImageNet-21k → ImageNet-1k
- **Размер входа**: 224×224 пикселей
- **Patch size**: 16×16
- **Количество параметров**: ~86M
- **Fine-tuning**: Полное дообучение всех слоев

**Обучение:**
- **Optimizer**: AdamW (lr=2e-5, weight_decay=0.01)
- **Scheduler**: Linear warmup + cosine decay
- **Batch size**: 16 (с gradient accumulation)
- **Epochs**: 3-5
- **Аугментации**: RandomHorizontalFlip, ColorJitter
- **Балансировка**: RandomOverSampler

## Внедрение

**Формат развертывания:** Микросервисная архитектура с поддержкой высоких нагрузок

**Компоненты системы:**

1. **ONNX модель** (`models/model.onnx`)
   - Универсальный формат для кроссплатформенного инференса
   - Размер: ~330 MB
   - Поддержка CPU/GPU

2. **Triton Inference Server**
   - Production-ready сервер инференса
   - REST/gRPC API
---


## Описание проекта

Проект реализует модель компьютерного зрения для классификации пневмонии на рентгеновских снимках грудной клетки. Система построена на базе Vision Transformer (ViT).

Основные возможности:

- Автоматическая загрузка и версионирование данных (DVC + Kaggle API).
- Управление конфигурациями через Hydra.
- Трекинг экспериментов и метрик (MLflow).
- Воспроизводимое окружение и управление зависимостями (Poetry).
- Подготовка к продакшену: экспорт в ONNX и скрипты для TensorRT / Triton Server.

## Данные

Используется датасет Chest X-Ray Images (Pneumonia).

- Вход: Рентгеновские снимки (JPEG).
- Классы: NORMAL (Здоров), PNEUMONIA (Пневмония).
- Препроцессинг: Ресайз до 224x224, нормализация (ImageNet stats), балансировка классов (RandomOverSampler).

## Setup

Инструкция для развертывания проекта на новой машине.

Предварительные требования

- Python 3.9+
- Poetry
- Аккаунт Kaggle (для скачивания данных)

Пошаговая установка

Клонирование репозитория и создание виртуального окружения:

```
git clone
cd pneumonia_detection
python3 -m venv venv
source venv/bin/activate
```

Установка зависимостей:

```
pip install poetry
poetry install
```

Настройка хуков качества кода (Pre-commit):

```
poetry run pre-commit install
```

Проверка: `poetry run pre-commit run -a`.

Настройка доступа к данным (Kaggle):
Проект автоматически скачивает данные. Для этого нужен API ключ.

Положите файл kaggle.json в `~/.kaggle/kaggle.json`.
Пример того, как выглядит файл:

```
{
  "username": "user_name",
  "key": "KGAT_123456789qwertyu"
}
```

## Train
Команда запуска:

```
poetry run python src/pneumonia_detect/commands.py train
```

Обучение
Конфигурация (Hydra)
Конфиги лежат в папке configs/.

Пример:

```
poetry run python src/pneumonia_detect/commands.py train training.num_train_epochs=5 model.learning_rate=1e-5
```

## Логирование (MLflow)

Метрики и параметры сохраняются в MLflow.
Чтобы увидеть графики, запустите сервер в отдельном терминале:

```
poetry run mlflow ui --port 8080
```

## Production Preparation

Подготовка модели к эксплуатации в высоконагруженных системах.

1. Конвертация в ONNX
   Перевод модели в универсальный формат для запуска без PyTorch.

```
poetry run python src/pneumonia_detect/to_onnx.py
```

Артефакт: `models/model.onnx`

2. Оптимизация TensorRT
   Для запуска на NVIDIA GPU (FP16 инференс).
   Скрипт использует Docker контейнер NVIDIA.

```
bash ./scripts/convert_to_trt.sh
```

Артефакт: `models/model.engine`

3. Triton Inference Server
   Подготовка Model Repository для запуска промышленного сервера инференса.

```bash
poetry run python src/pneumonia_detect/commands.py setup_triton
```

```bash
bash ./scripts/run_triton.sh
```

Проверка работы сервера:

```bash
curl localhost:8000/v2/health/ready

curl localhost:8000/v2/models/pneumonia_detection
```

## Infer

Скрипт предсказания поддерживает несколько бэкендов выполнения. Код инференса отделен от обучения.

Синтаксис:

```bash
poetry run python src/pneumonia_detect/commands.py infer <ПУТЬ_К_ИЗОБРАЖЕНИЮ> <РЕЖИМ>
```

Режимы работы:

- `onnx`: Использует onnxruntime и файл `models/model.onnx`
- `pytorch`: Использует оригинальные веса PyTorch.
- `triton`: Использует Triton Inference Server. Требует запущенный Triton сервер.
- `tensorrt`: Использует `models/model.engine`. Требует NVIDIA GPU.

Примеры запуска:

```bash
poetry run python src/pneumonia_detect/commands.py infer \
  "data/raw/chest_xray/chest_xray/test/NORMAL/NORMAL2-IM-0150-0001.jpeg" onnx


poetry run python src/pneumonia_detect/commands.py infer \
  "data/raw/chest_xray/chest_xray/test/PNEUMONIA/person1651_virus_2855.jpeg" triton
```

Формат вывода:

```json
{
  "filename": "person1651_virus_2855.jpeg",
  "mode": "triton",
  "prediction": "PNEUMONIA",
  "confidence": "0.9981"
}
```
