import logging
from pathlib import Path

import hydra
import torch
from omegaconf import DictConfig
from transformers import ViTForImageClassification, ViTImageProcessor

from pneumonia_detect import constants

log = logging.getLogger(__name__)


@hydra.main(config_path="../../configs", config_name="config", version_base="1.3")
def export_to_onnx(cfg: DictConfig):
    output_dir = Path(cfg.training.output_dir)
    final_model_path = output_dir / "final_model"
    onnx_path = Path(constants.MODELS_DIR) / "model.onnx"

    onnx_path.parent.mkdir(exist_ok=True, parents=True)

    if final_model_path.exists():
        model_id = str(final_model_path)
    else:
        log.info("Модель не найдена")
        return

    model = ViTForImageClassification.from_pretrained(
        model_id, attn_implementation=constants.ONNX_ATTENTION_IMPLEMENTATION
    )
    processor = ViTImageProcessor.from_pretrained(model_id)

    model.eval()

    height = processor.size["height"]
    width = processor.size["width"]
    dummy_input = torch.randn(1, constants.TRITON_INPUT_CHANNELS, height, width)

    torch.onnx.export(
        model,
        dummy_input,
        str(onnx_path),
        export_params=True,
        opset_version=constants.ONNX_OPSET_VERSION,
        do_constant_folding=True,
        input_names=[constants.ONNX_INPUT_NAME],
        output_names=[constants.ONNX_OUTPUT_NAME],
        dynamic_axes={
            constants.ONNX_INPUT_NAME: {0: "batch_size"},
            constants.ONNX_OUTPUT_NAME: {0: "batch_size"},
        },
    )

    log.info("Модель успешно экспортирована в ONNX")
    log.info(f"Файл: {onnx_path}")


if __name__ == "__main__":
    export_to_onnx()
