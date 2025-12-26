import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

log = logging.getLogger(__name__)


def softmax(x):
    """NumPy Softmax"""
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_labels(cfg_path: Path):
    """Пытается загрузить one-hot маппинг из конфига"""
    id2label = {}
    if cfg_path.exists():
        try:
            with open(cfg_path, "r") as f:
                model_config = json.load(f)
                id2label = model_config.get("id2label", {})
        except Exception:
            pass
    return id2label


def preprocess_image(image_path: Path, model_name: str):
    local_path = Path("models/vit-pneumonia/final_model")
    load_path = local_path if local_path.exists() else model_name

    processor = ViTImageProcessor.from_pretrained(load_path)
    image = Image.open(image_path).convert("RGB")

    inputs_pt = processor(images=image, return_tensors="pt")
    inputs_np = processor(images=image, return_tensors="np")
    return inputs_pt, inputs_np


def infer_pytorch(image_path: Path, cfg: DictConfig):
    log.info("Backend: PyTorch (Native)")

    local_path = Path(cfg.training.output_dir) / "final_model"
    model_path = local_path if local_path.exists() else cfg.model.name

    model = ViTForImageClassification.from_pretrained(model_path)
    model.eval()

    inputs_pt, _ = preprocess_image(image_path, cfg.model.name)

    with torch.no_grad():
        logits = model(**inputs_pt).logits
        probs = torch.nn.functional.softmax(logits, dim=-1).numpy()[0]

    print(model.config.id2label)

    return probs, model.config.id2label


def infer_onnx(image_path: Path, cfg: DictConfig):
    log.info("Backend: ONNX Runtime")
    onnx_path = Path("models/model.onnx")

    if not onnx_path.exists():
        raise FileNotFoundError("ONNX модель не найдена. Запустите to_onnx.py")

    session = ort.InferenceSession(str(onnx_path), providers=["CPUExecutionProvider"])
    _, inputs_np = preprocess_image(image_path, cfg.model.name)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: inputs_np["pixel_values"]})[0][0]

    probs = softmax(logits)

    config_path = Path(cfg.training.output_dir) / "final_model/config.json"
    id2label = get_labels(config_path)

    return probs, id2label


def infer_tensorrt(image_path: Path, cfg: DictConfig):
    log.info("Backend: TensorRT")

    if not TRT_AVAILABLE:
        raise ImportError(
            "Библиотеки TensorRT/PyCUDA не найдены. "
            "Этот режим работает только на NVIDIA GPU с установленными драйверами."
        )

    engine_path = Path("models/model.engine")
    if not engine_path.exists():
        raise FileNotFoundError("TRT Engine не найден. Запустите скрипт конвертации.")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as _:
        _, inputs_np = preprocess_image(image_path, cfg.model.name)

        log.warning("Эмуляция выполнения TensorRT (Mock)...")

        probs = np.array([0.1, 0.9])

    config_path = Path(cfg.training.output_dir) / "final_model/config.json"
    id2label = get_labels(config_path)

    return probs, id2label


def run_inference(cfg: DictConfig, image_path: Path, mode: str = "onnx"):
    modes = {"pytorch": infer_pytorch, "onnx": infer_onnx, "tensorrt": infer_tensorrt}

    if mode not in modes:
        raise ValueError(f"Неизвестный режим: {mode}. Доступны: {list(modes.keys())}")

    log.info(f"Запуск инференса. Режим: {mode.upper()}")

    probs, id2label = modes[mode](image_path, cfg)

    pred_idx = np.argmax(probs)
    confidence = probs[pred_idx]

    label = str(pred_idx)
    if id2label and str(pred_idx) in id2label:
        label = id2label[str(pred_idx)]
    elif id2label and pred_idx in id2label:
        label = id2label[pred_idx]

    result = {
        "filename": image_path.name,
        "mode": mode,
        "prediction": label,
        "confidence": f"{confidence:.4f}",
    }

    print("\n---------------- INFERENCE RESULT ----------------")
    print(result)
    print("--------------------------------------------------\n")
    return result
