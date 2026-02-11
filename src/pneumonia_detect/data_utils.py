import logging
import subprocess
import zipfile
from pathlib import Path

from pneumonia_detect import constants

log = logging.getLogger(__name__)


def download_data(dataset_path: Path) -> None:
    if dataset_path.exists() and any(dataset_path.iterdir()):
        log.info(f"Данные уже существуют в: {dataset_path}")
        return

    log.info(f"Данные не найдены в {dataset_path}. Начинаем загрузку...")

    dataset_path.parent.mkdir(parents=True, exist_ok=True)

    kaggle_dataset_name = constants.KAGGLE_DATASET_NAME

    try:
        subprocess.run(
            [
                "kaggle",
                "datasets",
                "download",
                "-d",
                kaggle_dataset_name,
                "--unzip",
                "-p",
                str(dataset_path),
            ],
            check=True,
        )
        log.info("Загрузка завершена успешно.")

        # иногда не срабатывает unzip
        # поэтому добавлено дополнительно
        zip_files = list(dataset_path.glob("*.zip"))
        if zip_files:
            log.info("Найден zip файл, выполняем распаковку...")
            for zip_file in zip_files:
                try:
                    with zipfile.ZipFile(zip_file, "r") as zip_ref:
                        zip_ref.extractall(dataset_path)
                    log.info(f"Архив {zip_file.name} успешно распакован.")
                    zip_file.unlink()
                    log.info(f"Архив {zip_file.name} удален.")
                except zipfile.BadZipFile:
                    log.error(f"Поврежденный zip файл: {zip_file}")
                    raise
        else:
            log.info("Распаковка завершена успешно.")

        _cleanup_macos_files(dataset_path)

    except FileNotFoundError:
        log.error(constants.ERROR_KAGGLE_CLI_NOT_FOUND)
        raise
    except subprocess.CalledProcessError as err:
        log.error(f"Ошибка при загрузке данных с Kaggle: {err}")
        raise


def _cleanup_macos_files(dataset_path: Path) -> None:
    import shutil

    macos_dir = dataset_path / constants.MACOS_HIDDEN_DIR
    if macos_dir.exists():
        shutil.rmtree(macos_dir)

    for file_path in dataset_path.rglob(constants.MACOS_HIDDEN_FILE_PREFIX + "*"):
        if file_path.is_file():
            file_path.unlink()

    for file_path in dataset_path.rglob(constants.MACOS_DS_STORE):
        if file_path.is_file():
            file_path.unlink()


def get_dvc_file(dataset_path: Path) -> Path:
    return dataset_path.parent / (dataset_path.name + ".dvc")
