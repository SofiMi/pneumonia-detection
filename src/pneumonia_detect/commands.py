import logging
import sys
from pathlib import Path

import fire
from dvc.repo import Repo
from hydra import compose, initialize

logging.basicConfig(level=logging.INFO)
log = logging.getLogger(__name__)


def _prepare_data(cfg) -> None:
    dataset_path = Path(cfg.data.dataset_path)

    from pneumonia_detect.data_utils import download_data

    download_data(dataset_path)

    log.info("Фиксация данных в DVC...")
    try:
        repo_root = Path(__file__).resolve().parent.parent.parent
        repo = Repo(repo_root)
        repo.add(str(dataset_path))
        log.info("Данные зафиксированы в .dvc файле")
    except Exception as err:
        log.warning(f"Не удалось выполнить DVC add: {err}")


def train(config_name: str = "config"):
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    _prepare_data(cfg)

    from pneumonia_detect.train import train_model

    log.info(f"Запуск тренировки с конфигом: {config_name}")
    train_model(cfg)


def infer(image_path: str, mode: str = "onnx", config_name: str = "config"):
    """
    mode: "pytorch", "onnx", "tensorrt"
    """
    with initialize(version_base="1.3", config_path="../../configs"):
        cfg = compose(config_name=config_name)

    image_file = Path(image_path)
    if not image_file.exists():
        log.error(f"Файл не найден: {image_file}")
        sys.exit(1)

    from pneumonia_detect.infer import run_inference

    run_inference(cfg, image_file, mode=mode)


if __name__ == "__main__":
    fire.Fire()
