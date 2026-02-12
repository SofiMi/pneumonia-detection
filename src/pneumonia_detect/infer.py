import json
import logging
from pathlib import Path

import numpy as np
import onnxruntime as ort
import torch
from omegaconf import DictConfig
from PIL import Image
from transformers import ViTForImageClassification, ViTImageProcessor

from pneumonia_detect import constants

try:
    import tensorrt as trt

    TRT_AVAILABLE = True
except ImportError:
    TRT_AVAILABLE = False

log = logging.getLogger(__name__)


def softmax(x):
    e_x = np.exp(x - np.max(x))
    return e_x / e_x.sum()


def get_labels(cfg_path: Path):
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
    local_path = Path(constants.DEFAULT_VIT_MODEL_PATH)
    load_path = local_path if local_path.exists() else model_name

    processor = ViTImageProcessor.from_pretrained(load_path)
    image = Image.open(image_path).convert(constants.IMAGE_MODE_RGB)

    inputs_pt = processor(images=image, return_tensors="pt")
    inputs_np = processor(images=image, return_tensors="np")
    return inputs_pt, inputs_np


def infer_pytorch(image_path: Path, cfg: DictConfig):
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
    onnx_path = Path(constants.ONNX_MODEL_PATH)

    if not onnx_path.exists():
        raise FileNotFoundError(constants.ERROR_ONNX_NOT_FOUND)

    session = ort.InferenceSession(
        str(onnx_path), providers=constants.DEFAULT_ONNX_PROVIDERS
    )
    _, inputs_np = preprocess_image(image_path, cfg.model.name)

    input_name = session.get_inputs()[0].name
    logits = session.run(None, {input_name: inputs_np["pixel_values"]})[0][0]

    probs = softmax(logits)

    config_path = (
        Path(cfg.training.output_dir) / "final_model" / constants.CONFIG_JSON_PATH
    )
    id2label = get_labels(config_path)

    return probs, id2label


def infer_tensorrt(image_path: Path, cfg: DictConfig):

    if not TRT_AVAILABLE:
        raise ImportError(constants.ERROR_TENSORRT_NOT_AVAILABLE)

    engine_path = Path(constants.TENSORRT_ENGINE_PATH)
    if not engine_path.exists():
        raise FileNotFoundError(constants.ERROR_TENSORRT_ENGINE_NOT_FOUND)

    try:
        import pycuda.driver as cuda

        use_gpu = True
        log.info("Используется GPU для TensorRT инференса")
    except ImportError:
        use_gpu = False
        log.info("PyCUDA недоступен, используется CPU режим для TensorRT")

    logger = trt.Logger(trt.Logger.WARNING)
    with open(engine_path, "rb") as f, trt.Runtime(logger) as runtime:
        engine = runtime.deserialize_cuda_engine(f.read())

    with engine.create_execution_context() as context:
        _, inputs_np = preprocess_image(image_path, cfg.model.name)
        input_data = inputs_np["pixel_values"].astype(np.float32)

        if use_gpu:
            output_data = np.empty((1, 2), dtype=np.float32)

            d_input = cuda.mem_alloc(input_data.nbytes)
            d_output = cuda.mem_alloc(output_data.nbytes)

            cuda.memcpy_htod(d_input, input_data)
            bindings = [int(d_input), int(d_output)]

            context.execute_v2(bindings)
            cuda.memcpy_dtoh(output_data, d_output)

            logits = output_data[0]
        else:
            output_data = np.empty((1, 2), dtype=np.float32)

            bindings = [input_data.ctypes.data, output_data.ctypes.data]

            context.execute_v2(bindings)
            logits = output_data[0]

        probs = softmax(logits)

    config_path = (
        Path(cfg.training.output_dir) / "final_model" / constants.CONFIG_JSON_PATH
    )
    id2label = get_labels(config_path)

    return probs, id2label


def infer_triton(image_path: Path, cfg: DictConfig):
    import urllib.request

    _, inputs_np = preprocess_image(image_path, cfg.model.name)

    pixel_values = inputs_np["pixel_values"]
    data = {
        "inputs": [
            {
                "name": constants.TRITON_INPUT_NAME,
                "shape": list(pixel_values.shape),
                "datatype": constants.TRITON_OUTPUT_DATATYPE,
                "data": pixel_values.flatten().tolist(),
            }
        ]
    }

    try:
        url = (
            f"http://{constants.DEFAULT_TRITON_HOST}:{constants.DEFAULT_TRITON_PORT}/"
            f"{constants.TRITON_API_VERSION}/models/{constants.TRITON_MODEL_NAME}/infer"
        )

        headers = {"Content-Type": "application/json"}

        json_data = json.dumps(data).encode("utf-8")
        req = urllib.request.Request(
            url, data=json_data, headers=headers, method="POST"
        )

        with urllib.request.urlopen(req) as response:
            result = json.loads(response.read().decode("utf-8"))

        output_data = result["outputs"][0]["data"]
        logits = np.array(output_data).reshape(-1, constants.TRITON_OUTPUT_CLASSES)[0]
        probs = softmax(logits)

        config_path = (
            Path(cfg.training.output_dir) / "final_model" / constants.CONFIG_JSON_PATH
        )
        id2label = get_labels(config_path)

        return probs, id2label

    except urllib.error.URLError:
        raise ConnectionError(constants.ERROR_TRITON_CONNECTION)
    except Exception as e:
        raise RuntimeError(f"Ошибка Triton инференса: {e}")


def run_inference(cfg: DictConfig, image_path: Path, mode: str = "onnx"):
    modes = {
        constants.SUPPORTED_MODES[0]: infer_pytorch,
        constants.SUPPORTED_MODES[1]: infer_onnx,
        constants.SUPPORTED_MODES[2]: infer_tensorrt,
        constants.SUPPORTED_MODES[3]: infer_triton,
    }

    if mode not in constants.SUPPORTED_MODES:
        raise ValueError(
            f"Неизвестный режим: {mode}. Доступны: {constants.SUPPORTED_MODES}"
        )

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
        "confidence": f"{confidence:.{constants.CONFIDENCE_DECIMAL_PLACES}f}",
    }

    print(result)
    return result
