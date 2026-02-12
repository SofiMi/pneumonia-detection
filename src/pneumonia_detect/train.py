import logging
import random
import subprocess
from pathlib import Path
from typing import List

import evaluate
import matplotlib.pyplot as plt
import mlflow
import pandas as pd
import torch
from datasets import ClassLabel, Dataset, Image
from imblearn.over_sampling import RandomOverSampler
from omegaconf import DictConfig, OmegaConf
from PIL import ImageFile
from torchvision.transforms import (
    Compose,
    Normalize,
    RandomAdjustSharpness,
    RandomRotation,
    Resize,
    ToTensor,
)
from transformers import (
    Trainer,
    TrainerCallback,
    TrainingArguments,
    ViTForImageClassification,
    ViTImageProcessor,
)

from pneumonia_detect import constants

ImageFile.LOAD_TRUNCATED_IMAGES = True

log = logging.getLogger(__name__)


def get_git_commit_id() -> str:
    try:
        commit_hash = (
            subprocess.check_output(["git", "rev-parse", "--short", "HEAD"])
            .strip()
            .decode("utf-8")
        )
        return commit_hash
    except Exception:
        log.warning(constants.ERROR_GIT_COMMIT_WARNING)
        return constants.GIT_COMMIT_UNKNOWN


def save_static_plots_from_mlflow(run_id: str):
    plots_dir = Path(constants.PLOTS_DIR)
    plots_dir.mkdir(exist_ok=True)

    try:
        client = mlflow.tracking.MlflowClient()
        client.get_run(run_id)

        metrics_history = {}
        for metric_name in [
            constants.METRIC_TRAIN_LOSS,
            constants.METRIC_EVAL_LOSS,
            constants.METRIC_EVAL_ACCURACY,
        ]:
            try:
                history = client.get_metric_history(run_id, metric_name)
                if history:
                    metrics_history[metric_name] = [(m.step, m.value) for m in history]
                    log.info(f"Найдено {len(history)} точек для {metric_name}")
            except Exception as e:
                log.warning(f"Не удалось получить историю для {metric_name}: {e}")

        fig, axes = plt.subplots(3, 1, figsize=(12, 15))
        fig.suptitle("Training Metrics", fontsize=16, fontweight="bold")

        # Train Loss
        if (
            constants.METRIC_TRAIN_LOSS in metrics_history
            and metrics_history[constants.METRIC_TRAIN_LOSS]
        ):
            data = [
                (step, value)
                for step, value in metrics_history[constants.METRIC_TRAIN_LOSS]
                if step > 0 and value > 0
            ]
            if data:
                steps, values = zip(*data)
                axes[0].plot(
                    steps,
                    values,
                    label="Train Loss",
                    color=constants.COLOR_BLUE,
                    linewidth=2,
                    marker="o",
                )
                axes[0].set_xlabel("Step")
                axes[0].set_ylabel("Loss")
                axes[0].set_title("Training Loss")
                axes[0].legend()
                axes[0].grid(True)

        # Validation Loss
        if (
            constants.METRIC_EVAL_LOSS in metrics_history
            and metrics_history[constants.METRIC_EVAL_LOSS]
        ):
            data = [
                (step, value)
                for step, value in metrics_history[constants.METRIC_EVAL_LOSS]
                if step > 0 and value > 0
            ]
            if data:
                steps, values = zip(*data)
                axes[1].plot(
                    steps,
                    values,
                    label="Eval Loss",
                    color=constants.COLOR_RED,
                    linewidth=2,
                    marker="o",
                )
                axes[1].set_xlabel("Step")
                axes[1].set_ylabel("Loss")
                axes[1].set_title("Validation Loss")
                axes[1].legend()
                axes[1].grid(True)

        # Validation Accuracy
        if (
            constants.METRIC_EVAL_ACCURACY in metrics_history
            and metrics_history[constants.METRIC_EVAL_ACCURACY]
        ):
            data = [
                (step, value)
                for step, value in metrics_history[constants.METRIC_EVAL_ACCURACY]
                if step > 0 and value >= 0
            ]
            if data:
                steps, values = zip(*data)
                axes[2].plot(
                    steps,
                    values,
                    label="Accuracy",
                    color=constants.COLOR_GREEN,
                    linewidth=2,
                    marker="o",
                )
                axes[2].set_xlabel("Step")
                axes[2].set_ylabel("Accuracy")
                axes[2].set_title("Validation Accuracy")
                axes[2].set_ylim(0, 1)
                axes[2].legend()
                axes[2].grid(True)

        plt.tight_layout()
        plt.savefig(
            plots_dir / "training_metrics.png",
            dpi=constants.DPI_SETTING,
            bbox_inches="tight",
        )
        plt.close()

    except Exception as e:
        log.error(f"Ошибка при создании графиков из MLflow: {e}")


def prepare_dataframe(
    dataset_path: str,
    random_state: int = constants.DEFAULT_RANDOM_SEED,
    max_samples: int = None,
) -> pd.DataFrame:
    base_path = Path(dataset_path)
    file_names: List[str] = []
    labels: List[str] = []

    train_patterns = constants.TRAIN_IMAGE_PATTERNS
    found_files = []
    for pattern in train_patterns:
        found_files.extend(list(base_path.rglob(pattern)))

    if not found_files:
        raise FileNotFoundError(constants.ERROR_IMAGES_NOT_FOUND.format(dataset_path))

    log.info(f"Найдено {len(found_files)} файлов в папке train.")

    random.seed(random_state)
    half_count = len(found_files) // constants.DATA_REDUCTION_FACTOR
    found_files = random.sample(found_files, half_count)
    log.info(
        "Используется 1/%s часть данных: %s файлов для быстрого обучения.",
        constants.DATA_REDUCTION_FACTOR,
        len(found_files),
    )

    for file_path in found_files:
        label = file_path.parent.name
        labels.append(label)
        file_names.append(str(file_path))

    df = pd.DataFrame.from_dict({"image": file_names, "label": labels})

    y = df[["label"]]
    df_features = df.drop(["label"], axis=1)
    ros = RandomOverSampler(random_state=random_state)
    df_resampled, y_resampled = ros.fit_resample(df_features, y)
    df_resampled["label"] = y_resampled

    return df_resampled


def get_transforms(processor):
    image_mean, image_std = processor.image_mean, processor.image_std
    size = processor.size["height"]
    normalize = Normalize(mean=image_mean, std=image_std)

    _train_transforms = Compose(
        [
            Resize((size, size)),
            RandomRotation(constants.PLOT_ROTATION_DEGREES),
            RandomAdjustSharpness(constants.PLOT_SHARPNESS_FACTOR),
            ToTensor(),
            normalize,
        ]
    )

    _val_transforms = Compose(
        [
            Resize((size, size)),
            ToTensor(),
            normalize,
        ]
    )

    def train_transforms(examples):
        examples["pixel_values"] = [
            _train_transforms(image.convert(constants.IMAGE_MODE_RGB))
            for image in examples["image"]
        ]
        return examples

    def val_transforms(examples):
        examples["pixel_values"] = [
            _val_transforms(image.convert(constants.IMAGE_MODE_RGB))
            for image in examples["image"]
        ]
        return examples

    return train_transforms, val_transforms


def compute_metrics(eval_pred):
    accuracy = evaluate.load(constants.METRIC_ACCURACY)
    predictions = eval_pred.predictions
    label_ids = eval_pred.label_ids
    predicted_labels = predictions.argmax(axis=1)
    acc_score = accuracy.compute(predictions=predicted_labels, references=label_ids)[
        constants.METRIC_ACCURACY
    ]
    return {constants.METRIC_ACCURACY: acc_score}


class TrainLossLoggingCallback(TrainerCallback):
    def on_log(self, args, state, control, logs=None, **kwargs):
        if logs is not None and "loss" in logs:
            mlflow.log_metric("train_loss", logs["loss"], step=state.global_step)


def collate_fn(examples):
    pixel_values = torch.stack([example["pixel_values"] for example in examples])
    labels = torch.tensor([example["label"] for example in examples])
    return {"pixel_values": pixel_values, "labels": labels}


def train_model(cfg: DictConfig):
    mlflow.set_tracking_uri(cfg.mlflow.tracking_uri)
    mlflow.set_experiment(cfg.mlflow.experiment_name)

    git_commit_id = get_git_commit_id()

    with mlflow.start_run(run_name=f"train_{git_commit_id}") as run:
        log.info(f"MLflow Run ID: {run.info.run_id}")

        mlflow.log_params(OmegaConf.to_container(cfg, resolve=True))
        mlflow.set_tag("git_commit_id", git_commit_id)

        log.info(f"Загрузка данных из {cfg.data.dataset_path}")
        df = prepare_dataframe(cfg.data.dataset_path, cfg.random_seed)

        dataset = Dataset.from_pandas(df).cast_column("image", Image())
        labels_list = sorted(list(set(df["label"].unique())))
        label2id = {label: i for i, label in enumerate(labels_list)}
        id2label = {i: label for i, label in enumerate(labels_list)}

        ClassLabels = ClassLabel(num_classes=len(labels_list), names=labels_list)

        def map_label2id(example):
            example["label"] = ClassLabels.str2int(example["label"])
            return example

        dataset = dataset.map(map_label2id, batched=True)
        dataset = dataset.cast_column("label", ClassLabels)

        split_dataset = dataset.train_test_split(
            test_size=cfg.data.test_size, shuffle=True, stratify_by_column="label"
        )
        train_data = split_dataset["train"]
        test_data = split_dataset["test"]

        model_name = cfg.model.name
        processor = ViTImageProcessor.from_pretrained(model_name)
        model = ViTForImageClassification.from_pretrained(
            model_name,
            num_labels=len(labels_list),
            id2label=id2label,
            label2id=label2id,
            ignore_mismatched_sizes=True,
        )

        train_trans_func, val_trans_func = get_transforms(processor)
        train_data.set_transform(train_trans_func)
        test_data.set_transform(val_trans_func)

        training_args = TrainingArguments(
            output_dir=cfg.training.output_dir,
            logging_dir=f"{cfg.training.output_dir}/logs",
            eval_strategy="steps",
            save_strategy="steps",
            learning_rate=cfg.model.learning_rate,
            per_device_train_batch_size=cfg.training.batch_size,
            per_device_eval_batch_size=cfg.training.batch_size,
            num_train_epochs=cfg.training.num_train_epochs,
            weight_decay=cfg.model.weight_decay,
            warmup_steps=cfg.training.warmup_steps,
            logging_steps=getattr(
                cfg.training, "logging_steps", constants.DEFAULT_LOGGING_STEPS
            ),
            eval_steps=getattr(
                cfg.training, "eval_steps", constants.DEFAULT_EVAL_STEPS
            ),
            save_steps=getattr(
                cfg.training, "eval_steps", constants.DEFAULT_EVAL_STEPS
            ),
            remove_unused_columns=False,
            save_total_limit=cfg.training.save_total_limit,
            load_best_model_at_end=True,
            metric_for_best_model=constants.METRIC_ACCURACY,
            report_to="mlflow",
            run_name=f"train_{git_commit_id}",
        )

        trainer = Trainer(
            model=model,
            args=training_args,
            train_dataset=train_data,
            eval_dataset=test_data,
            data_collator=collate_fn,
            compute_metrics=compute_metrics,
            tokenizer=processor,
            callbacks=[TrainLossLoggingCallback()],
        )

        log.info("Начало обучения")
        trainer.train()

        log.info("Финальная оценка")
        metrics = trainer.evaluate()
        log.info(f"Метрики: {metrics}")

        mlflow.log_metrics(metrics)

        save_path = f"{cfg.training.output_dir}/final_model"
        trainer.save_model(save_path)
        processor.save_pretrained(save_path)

        save_static_plots_from_mlflow(run.info.run_id)
