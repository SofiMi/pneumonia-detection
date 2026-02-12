import logging
import sys
from pathlib import Path

import fire
from dvc.repo import Repo
from hydra import compose, initialize

from pneumonia_detect import constants
from pneumonia_detect.data_utils import download_data
from pneumonia_detect.infer import run_inference
from pneumonia_detect.train import train_model
from pneumonia_detect.triton_setup import create_triton_model_repository

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _prepare_data(cfg) -> None:
    dataset_path = Path(cfg.data.dataset_path)
    download_data(dataset_path)
    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        repo = Repo(repo_root)
        repo.add(str(dataset_path))
    except Exception as err:
        log.warning(f"Не удалось выполнить DVC add: {err}")


def train(config_name: str = "config"):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    _prepare_data(cfg)

    log.info(f"Запуск тренировки с конфигом: {config_name}")
    train_model(cfg)


def infer(image_path: str, mode: str = "onnx", config_name: str = "config"):
    """
    mode: поддерживаемые режимы из constants.SUPPORTED_MODES
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    image_file = Path(image_path)
    if not image_file.exists():
        log.error(f"Файл не найден: {image_file}")
        sys.exit(1)

    if mode not in constants.SUPPORTED_MODES:
        log.error(
            f"Неподдерживаемый режим: {mode}. Доступны: {constants.SUPPORTED_MODES}"
        )
        sys.exit(1)

    run_inference(cfg, image_file, mode=mode)


def setup_triton(model_version: str = constants.DEFAULT_MODEL_VERSION):
    """
    Подготовка модели для Triton Inference Server.

    Args:
        model_version: Версия модели
    """

    if not create_triton_model_repository(model_version):
        log.error("Не удалось создать репозиторий моделей")


if __name__ == "__main__":
    fire.Fire()
