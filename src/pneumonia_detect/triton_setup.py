import logging
import shutil
from pathlib import Path

from . import constants

log = logging.getLogger(__name__)


def create_triton_model_repository(model_version: str = constants.DEFAULT_MODEL_VERSION):
    project_root = Path(__file__).parent.parent.parent
    model_repo = project_root / "triton_models"
    pneumonia_model_dir = model_repo / constants.TRITON_MODEL_NAME / model_version

    pneumonia_model_dir.mkdir(parents=True, exist_ok=True)

    onnx_source = project_root / constants.MODELS_DIR / "model.onnx"
    onnx_dest = pneumonia_model_dir / "model.onnx"

    if onnx_source.exists():
        shutil.copy2(onnx_source, onnx_dest)
        log.info(f"ONNX модель скопирована: {onnx_dest}")
    else:
        log.error(f"ONNX модель не найдена: {onnx_source}")
        return False

    config_content = constants.TRITON_CONFIG_TEMPLATE.format(
        max_batch_size=constants.TRITON_MAX_BATCH_SIZE,
        input_name=constants.TRITON_INPUT_NAME,
        channels=constants.TRITON_INPUT_CHANNELS,
        height=constants.TRITON_INPUT_HEIGHT,
        width=constants.TRITON_INPUT_WIDTH,
        output_name=constants.ONNX_OUTPUT_NAME,
        output_classes=constants.TRITON_OUTPUT_CLASSES,
        instance_count=constants.TRITON_INSTANCE_COUNT,
        max_queue_delay=constants.TRITON_MAX_QUEUE_DELAY_MICROSECONDS
    )

    config_path = model_repo / constants.TRITON_MODEL_NAME / "config.pbtxt"
    config_path.write_text(config_content)
    log.info(f"Конфигурация Triton создана: {config_path}")

    labels_content = "\n".join(constants.TRITON_LABELS)
    labels_path = model_repo / constants.TRITON_MODEL_NAME / "labels.txt"
    labels_path.write_text(labels_content)
    log.info(f"Файл меток создан: {labels_path}")

    log.info("Подготовка к Triton завершена успешно!")
    log.info("Структура создана:")
    log.info(f"  {pneumonia_model_dir / 'model.onnx'}")
    log.info(f"  {config_path}")
    log.info(f"  {labels_path}")

    log.info(f"\nДля запуска Triton сервера используйте:")
    log.info(f"  docker run --rm -p{constants.DEFAULT_TRITON_PORT}:{constants.DEFAULT_TRITON_PORT} -p8001:8001 -p8002:8002 \\")
    log.info("    -v $(pwd)/triton_models:/models \\")
    log.info("    nvcr.io/nvidia/tritonserver:23.10-py3 \\")
    log.info("    tritonserver --model-repository=/models")

    return True
