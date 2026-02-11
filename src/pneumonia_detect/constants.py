# Model paths and directories
DEFAULT_VIT_MODEL_PATH = "models/vit-pneumonia/final_model"
ONNX_MODEL_PATH = "models/model.onnx"
TENSORRT_ENGINE_PATH = "models/model.engine"
CONFIG_JSON_PATH = "config.json"
MODELS_DIR = "models"
PLOTS_DIR = "plots"

# Execution providers
DEFAULT_ONNX_PROVIDERS = ["CPUExecutionProvider"]

# TensorRT mock probabilities (placeholder for actual inference)
TENSORRT_MOCK_PROBABILITIES = [0.1, 0.9]

# Formatting and precision
CONFIDENCE_DECIMAL_PLACES = 4
DPI_SETTING = 150
FIGURE_SIZE = (10, 6)
FIGURE_SIZE_SMALL = (8, 6)

# Triton server configuration
DEFAULT_TRITON_HOST = "localhost"
DEFAULT_TRITON_PORT = 8000
TRITON_MODEL_NAME = "pneumonia_detection"
TRITON_API_VERSION = "v2"
TRITON_INPUT_NAME = "input"
TRITON_OUTPUT_DATATYPE = "FP32"
TRITON_MAX_BATCH_SIZE = 8
TRITON_MAX_QUEUE_DELAY_MICROSECONDS = 100

# Model configuration
TRITON_INPUT_CHANNELS = 3
TRITON_INPUT_HEIGHT = 224
TRITON_INPUT_WIDTH = 224
TRITON_OUTPUT_CLASSES = 2
TRITON_INSTANCE_COUNT = 1

# ONNX export settings
ONNX_OPSET_VERSION = 12
ONNX_INPUT_NAME = "input"
ONNX_OUTPUT_NAME = "output"
ONNX_ATTENTION_IMPLEMENTATION = "eager"

# Training constants
DEFAULT_RANDOM_SEED = 42
DATA_REDUCTION_FACTOR = 15  # Use 1/15 of data for fast training
PLOT_ROTATION_DEGREES = 20
PLOT_SHARPNESS_FACTOR = 2
DEFAULT_LOGGING_STEPS = 500
DEFAULT_EVAL_STEPS = 500
GIT_COMMIT_UNKNOWN = "unknown"
IMAGE_MODE_RGB = "RGB"

# Data processing
KAGGLE_DATASET_NAME = "paultimothymooney/chest-xray-pneumonia"
MACOS_HIDDEN_DIR = "__MACOSX"
MACOS_HIDDEN_FILE_PREFIX = "._"
MACOS_DS_STORE = ".DS_Store"

# File patterns for data loading
TRAIN_IMAGE_PATTERNS = ["*/train/*/*.jpeg", "*/train/*/*.jpg", "*/train/*/*.png"]

# Triton model labels
TRITON_LABELS = ["NORMAL", "PNEUMONIA"]

# Default model version
DEFAULT_MODEL_VERSION = "1"

# Error messages
ERROR_ONNX_NOT_FOUND = "ONNX модель не найдена. Запустите to_onnx.py"
ERROR_TENSORRT_NOT_AVAILABLE = (
    "Библиотеки TensorRT/PyCUDA не найдены. "
    "Этот режим работает только на NVIDIA GPU с установленными драйверами."
)
ERROR_TENSORRT_ENGINE_NOT_FOUND = "TRT Engine не найден. Запустите скрипт конвертации."
ERROR_TRITON_CONNECTION = (
    "Не удалось подключиться к Triton серверу. "
    "Убедитесь, что сервер запущен: ./scripts/run_triton.sh"
)
ERROR_KAGGLE_CLI_NOT_FOUND = (
    "Kaggle CLI не найден. Убедитесь, что 'kaggle' установлен "
    "и файл kaggle.json находится в ~/.kaggle/"
)
ERROR_GIT_COMMIT_WARNING = "Не удалось получить Git commit ID. Используем 'unknown'."
ERROR_IMAGES_NOT_FOUND = "Изображения не найдены в папке train в {}"

# Supported inference modes
SUPPORTED_MODES = ["pytorch", "onnx", "tensorrt", "triton"]

# Metrics names
METRIC_TRAIN_LOSS = "train_loss"
METRIC_EVAL_LOSS = "eval_loss"
METRIC_EVAL_ACCURACY = "eval_accuracy"
METRIC_ACCURACY = "accuracy"

# Plot colors
COLOR_BLUE = "blue"
COLOR_RED = "red"
COLOR_GREEN = "green"

# Triton configuration template
TRITON_CONFIG_TEMPLATE = """name: "pneumonia_detection"
platform: "onnxruntime_onnx"
max_batch_size: {max_batch_size}
input [
  {{
    name: "{input_name}"
    data_type: TYPE_FP32
    dims: [ {channels}, {height}, {width} ]
  }}
]
output [
  {{
    name: "{output_name}"
    data_type: TYPE_FP32
    dims: [ {output_classes} ]
  }}
]
instance_group [
  {{
    count: {instance_count}
    kind: KIND_CPU
  }}
]
dynamic_batching {{
  max_queue_delay_microseconds: {max_queue_delay}
}}"""
