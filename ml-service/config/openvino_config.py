"""
OpenVINO Configuration - Model paths, device settings, and inference parameters
"""

from pathlib import Path
from enum import Enum

# ============================================================================
# BASE PATHS
# ============================================================================

BASE_DIR = Path(__file__).resolve().parent.parent
MODELS_DIR = BASE_DIR / "openvino_models"
CACHE_DIR = BASE_DIR / "cache"
LOGS_DIR = BASE_DIR.parent / "logs"

# Create directories if they don't exist
MODELS_DIR.mkdir(parents=True, exist_ok=True)
CACHE_DIR.mkdir(parents=True, exist_ok=True)
LOGS_DIR.mkdir(parents=True, exist_ok=True)

# ============================================================================
# DEVICE CONFIGURATION
# ============================================================================

class DeviceType(str, Enum):
    CPU = "CPU"
    GPU = "GPU"
    AUTO = "AUTO"

DEVICE_CONFIG = {
    "primary_device": DeviceType.AUTO.value,
    "fallback_device": DeviceType.CPU.value,
    "enable_gpu": True,
    "enable_threading": True,
    "num_streams": 1,
    "num_threads": 4,
}

# ============================================================================
# MODEL PATHS
# ============================================================================

MODEL_PATHS = {
    "face_detector": {
        "model_type": "face-detection",
        "xml": str(MODELS_DIR / "face-detection" / "face-detection-0206.xml"),
        "bin": str(MODELS_DIR / "face-detection" / "face-detection-0206.bin"),
        "input_size": (384, 672),
        "description": "Ultra-lightweight face detection model"
    },
    
    "object_detector": {
        "model_type": "object-detection",
        "xml": str(MODELS_DIR / "object-detection" / "yolov8n.xml"),
        "bin": str(MODELS_DIR / "object-detection" / "yolov8n.bin"),
        "input_size": (640, 640),
        "description": "YOLOv8 nano - lightweight object detection"
    },
    
    "nsfw_detector": {
        "model_type": "nsfw-detection",
        "xml": str(MODELS_DIR / "nsfw-detection" / "mobilenet-nsfw.xml"),
        "bin": str(MODELS_DIR / "nsfw-detection" / "mobilenet-nsfw.bin"),
        "input_size": (224, 224),
        "description": "NSFW content detection"
    },
    
    "age_gender_recognizer": {
        "model_type": "age-gender",
        "xml": str(MODELS_DIR / "age-gender" / "age-gender-recognition-retail-0013.xml"),
        "bin": str(MODELS_DIR / "age-gender" / "age-gender-recognition-retail-0013.bin"),
        "input_size": (62, 62),
        "description": "Age and gender recognition model"
    },
    
    "emotion_recognizer": {
        "model_type": "emotion-recognition",
        "xml": str(MODELS_DIR / "emotion-recognition" / "emotions-recognition-retail-0003.xml"),
        "bin": str(MODELS_DIR / "emotion-recognition" / "emotions-recognition-retail-0003.bin"),
        "input_size": (64, 64),
        "description": "Emotion recognition from face images"
    },
}

# ============================================================================
# CONFIDENCE THRESHOLDS
# ============================================================================

CONFIDENCE_THRESHOLDS = {
    "face_detector": 0.5,
    "object_detector": 0.45,
    "age_gender_recognizer": 0.5,
    "emotion_recognizer": 0.5,
    "nsfw_detector": 0.6,
}

# ============================================================================
# SUPPORTED FORMATS
# ============================================================================

SUPPORTED_FORMATS = {
    "images": [
        "image/jpeg",
        "image/png",
        "image/bmp",
        "image/webp",
    ],
    "videos": [
        "video/mp4",
        "video/x-msvideo",
        "video/quicktime",
        "video/x-matroska"
    ]
}

# ============================================================================
# VIDEO PROCESSING SETTINGS
# ============================================================================

VIDEO_SETTINGS = {
    "max_frame_size": (1920, 1080),
    "default_sample_rate": 5,
    "max_video_duration_seconds": 600,
    "max_file_size_mb": 500,
}

# ============================================================================
# PREPROCESSING SETTINGS
# ============================================================================

PREPROCESSING = {
    "normalize": True,
    "normalize_mean": [0.485, 0.456, 0.406],
    "normalize_std": [0.229, 0.224, 0.225],
    "swap_rb": True,
    "auto_resize": True,
}

# ============================================================================
# INFERENCE OPTIMIZATION SETTINGS
# ============================================================================

INFERENCE_CONFIG = {
    "use_async_infer": True,
    "batch_size": 1,
    "compile_preprocess": True,
    "use_cache": True,
    "cache_dir": str(CACHE_DIR),
}

# ============================================================================
# LOGGING CONFIGURATION
# ============================================================================

LOGGING = {
    "level": "INFO",
    "format": "%(asctime)s - %(name)s - %(levelname)s - %(message)s",
    "log_file": str(LOGS_DIR / "openvino_service.log"),
}

# ============================================================================
# UTILITY FUNCTIONS
# ============================================================================

def get_model_path(model_name: str) -> dict:
    """Get model paths"""
    if model_name not in MODEL_PATHS:
        raise ValueError(f"Model {model_name} not found in configuration")
    
    return MODEL_PATHS[model_name]

def get_available_devices():
    """Returns list of available devices"""
    try:
        from openvino.runtime import Core
        ie = Core()
        return ie.available_devices
    except Exception as e:
        return ["CPU"]

def validate_image_format(content_type: str) -> bool:
    """Validate image content type"""
    return content_type in SUPPORTED_FORMATS["images"]

def validate_video_format(content_type: str) -> bool:
    """Validate video content type"""
    return content_type in SUPPORTED_FORMATS["videos"]

def get_confidence_threshold(model_name: str) -> float:
    """Get confidence threshold for a model"""
    return CONFIDENCE_THRESHOLDS.get(model_name, 0.5)

def get_model_input_size(model_name: str) -> tuple:
    """Get expected input size for a model"""
    if model_name in MODEL_PATHS:
        return MODEL_PATHS[model_name].get("input_size", (640, 640))
    return (640, 640)