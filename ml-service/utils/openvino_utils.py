"""
OpenVINO Utilities - Model loading, preprocessing, and result formatting
"""

import cv2
import numpy as np
from pathlib import Path
from typing import Dict, Tuple, List, Any
import logging
import os

logger = logging.getLogger(__name__)

# ============================================================================
# MODEL LOADER
# ============================================================================

class ModelLoader:
    """Load and cache OpenVINO models"""
    
    def __init__(self, ie_core):
        """
        Initialize model loader
        
        Args:
            ie_core: OpenVINO IE.Core instance
        """
        self.ie = ie_core
        self.loaded_models = {}
    
    def load_model(self, model_name: str):
        """
        Load a model from configuration
        
        Args:
            model_name: Name of model from config
            
        Returns:
            Compiled model ready for inference
        """
        if model_name in self.loaded_models:
            return self.loaded_models[model_name]
        
        from config.openvino_config import MODEL_PATHS, DEVICE_CONFIG
        
        if model_name not in MODEL_PATHS:
            raise ValueError(f"Model {model_name} not found")
        
        model_config = MODEL_PATHS[model_name]
        xml_path = model_config["xml"]
        bin_path = model_config["bin"]
        
        # Check if model files exist
        if not os.path.exists(xml_path) or not os.path.exists(bin_path):
            logger.warning(f"Model files not found for {model_name}")
            raise FileNotFoundError(f"Model files not found: {xml_path}, {bin_path}")
        
        try:
            # Read and compile model
            logger.info(f"Loading model: {model_name}")
            ov_model = self.ie.read_model(model=xml_path)
            
            # Get primary device
            device = DEVICE_CONFIG["primary_device"]
            
            try:
                compiled_model = self.ie.compile_model(ov_model, device)
                logger.info(f"Model {model_name} compiled on {device}")
            except Exception as e:
                logger.warning(f"Failed to compile on {device}, falling back to CPU")
                compiled_model = self.ie.compile_model(ov_model, "CPU")
            
            self.loaded_models[model_name] = compiled_model
            return compiled_model
            
        except Exception as e:
            logger.error(f"Failed to load model {model_name}: {str(e)}")
            raise
    
    def unload_model(self, model_name: str):
        """Unload a model from memory"""
        if model_name in self.loaded_models:
            del self.loaded_models[model_name]
            logger.info(f"Unloaded model: {model_name}")

# ============================================================================
# IMAGE PREPROCESSOR
# ============================================================================

class ImagePreprocessor:
    """Preprocess images for OpenVINO inference"""
    
    def __init__(self, model_name: str):
        """
        Initialize preprocessor for specific model
        
        Args:
            model_name: Name of the model
        """
        from config.openvino_config import MODEL_PATHS, PREPROCESSING
        
        self.model_name = model_name
        self.model_config = MODEL_PATHS.get(model_name, {})
        self.preprocessing_config = PREPROCESSING
        self.target_size = self.model_config.get("input_size", (640, 640))
    
    def preprocess(self, image: np.ndarray) -> Tuple[np.ndarray, Dict]:
        """
        Preprocess image for inference
        
        Args:
            image: Input image (BGR format from OpenCV)
            
        Returns:
            Tuple of (preprocessed_image, scale_factors)
        """
        original_h, original_w = image.shape[:2]
        scale_factors = {
            "scale_x": self.target_size[1] / original_w,
            "scale_y": self.target_size[0] / original_h,
            "original_width": original_w,
            "original_height": original_h
        }
        
        # Resize
        resized = cv2.resize(image, self.target_size)
        
        # Convert BGR to RGB if needed
        if self.preprocessing_config.get("swap_rb", True):
            resized = cv2.cvtColor(resized, cv2.COLOR_BGR2RGB)
        
        # Transpose to NCHW format
        prepared = np.transpose(resized, (2, 0, 1))
        
        # Add batch dimension
        prepared = np.expand_dims(prepared, 0).astype(np.float32)
        
        # Normalize
        if self.preprocessing_config.get("normalize", True):
            mean = np.array(self.preprocessing_config["normalize_mean"])
            std = np.array(self.preprocessing_config["normalize_std"])
            
            mean = mean.reshape(1, 3, 1, 1)
            std = std.reshape(1, 3, 1, 1)
            
            prepared = (prepared / 255.0 - mean) / std
        else:
            prepared = prepared / 255.0
        
        return prepared, scale_factors

# ============================================================================
# VIDEO PROCESSOR
# ============================================================================

class VideoProcessor:
    """Process video files for frame-by-frame analysis"""
    
    def __init__(self, video_path: str):
        """
        Initialize video processor
        
        Args:
            video_path: Path to video file
        """
        self.video_path = video_path
        self.cap = cv2.VideoCapture(video_path)
        
        if not self.cap.isOpened():
            raise IOError(f"Failed to open video: {video_path}")
        
        self.total_frames = int(self.cap.get(cv2.CAP_PROP_FRAME_COUNT))
        self.fps = self.cap.get(cv2.CAP_PROP_FPS)
        self.width = int(self.cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        self.height = int(self.cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        self.current_frame = 0
    
    def get_frame(self, frame_index: int = None) -> Tuple[bool, np.ndarray]:
        """
        Get specific frame
        
        Args:
            frame_index: Frame number to retrieve (0-based)
            
        Returns:
            Tuple of (success, frame)
        """
        if frame_index is not None:
            self.cap.set(cv2.CAP_PROP_POS_FRAMES, frame_index)
        
        ret, frame = self.cap.read()
        if ret:
            self.current_frame = int(self.cap.get(cv2.CAP_PROP_POS_FRAMES))
        return ret, frame
    
    def get_frames_sample(self, sample_rate: int = 5) -> List[Tuple[int, np.ndarray]]:
        """
        Get sampled frames from video
        
        Args:
            sample_rate: Process every Nth frame
            
        Returns:
            List of (frame_number, frame) tuples
        """
        frames = []
        self.cap.set(cv2.CAP_PROP_POS_FRAMES, 0)
        frame_idx = 0
        
        while True:
            ret, frame = self.cap.read()
            if not ret:
                break
            
            if frame_idx % sample_rate == 0:
                frames.append((frame_idx, frame))
            
            frame_idx += 1
        
        return frames
    
    def get_video_info(self) -> Dict:
        """Get video information"""
        return {
            "path": self.video_path,
            "total_frames": self.total_frames,
            "fps": self.fps,
            "width": self.width,
            "height": self.height,
            "duration_seconds": self.total_frames / self.fps if self.fps > 0 else 0
        }
    
    def release(self):
        """Release video capture"""
        if self.cap.isOpened():
            self.cap.release()
    
    def __del__(self):
        """Cleanup on deletion"""
        self.release()

# ============================================================================
# RESULT FORMATTER
# ============================================================================

class ResultFormatter:
    """Format model inference results"""
    
    def __init__(self, model_name: str):
        """
        Initialize result formatter
        
        Args:
            model_name: Name of the model
        """
        from config.openvino_config import MODEL_PATHS, CONFIDENCE_THRESHOLDS
        
        self.model_name = model_name
        self.model_config = MODEL_PATHS.get(model_name, {})
        self.confidence_threshold = CONFIDENCE_THRESHOLDS.get(model_name, 0.5)
        self.model_type = self.model_config.get("model_type", "unknown")
    
    def format_output(
        self,
        output: np.ndarray,
        original_height: int,
        original_width: int,
        scale_factors: Dict = None
    ) -> Dict[str, Any]:
        """
        Format model output based on model type
        
        Args:
            output: Raw model output
            original_height: Original image height
            original_width: Original image width
            scale_factors: Scaling factors from preprocessing
            
        Returns:
            Formatted detection results
        """
        if self.model_type == "object-detection":
            return self._format_object_detection(output, original_height, original_width)
        
        elif self.model_type == "face-detection":
            return self._format_face_detection(output, original_height, original_width)
        
        elif self.model_type == "age-gender":
            return self._format_age_gender(output)
        
        elif self.model_type == "emotion-recognition":
            return self._format_emotion(output)
        
        elif self.model_type == "nsfw-detection":
            return self._format_nsfw(output)
        
        else:
            return {"detections": [], "model_type": self.model_type}
    
    def _format_object_detection(
        self,
        output: np.ndarray,
        original_height: int,
        original_width: int
    ) -> Dict:
        """Format object detection output"""
        detections = []
        
        # Flatten output if needed
        if len(output.shape) > 2:
            output = output.reshape(-1, output.shape[-1])
        
        # Process detections
        for detection in output:
            if len(detection) < 6:
                continue
            
            confidence = detection[4]
            
            if confidence < self.confidence_threshold:
                continue
            
            x_center = detection[0] * original_width
            y_center = detection[1] * original_height
            width = detection[2] * original_width
            height = detection[3] * original_height
            
            x1 = int(max(0, x_center - width / 2))
            y1 = int(max(0, y_center - height / 2))
            x2 = int(min(original_width, x_center + width / 2))
            y2 = int(min(original_height, y_center + height / 2))
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "area": (x2 - x1) * (y2 - y1)
            })
        
        return {
            "detections": sorted(detections, key=lambda x: x["confidence"], reverse=True),
            "num_detections": len(detections)
        }
    
    def _format_face_detection(
        self,
        output: np.ndarray,
        original_height: int,
        original_width: int
    ) -> Dict:
        """Format face detection output"""
        detections = []
        
        # Reshape output
        output = output.reshape(-1, 5)
        
        for detection in output:
            confidence = detection[4]
            if confidence < self.confidence_threshold:
                continue
            
            x1 = int(detection[0] * original_width)
            y1 = int(detection[1] * original_height)
            x2 = int(detection[2] * original_width)
            y2 = int(detection[3] * original_height)
            
            detections.append({
                "bbox": [x1, y1, x2, y2],
                "confidence": float(confidence),
                "width": x2 - x1,
                "height": y2 - y1
            })
        
        return {
            "detections": sorted(detections, key=lambda x: x["confidence"], reverse=True),
            "num_detections": len(detections)
        }
    
    def _format_age_gender(self, output: np.ndarray) -> Dict:
        """Format age/gender recognition output"""
        age_preds = output[0][0:100]
        gender_preds = output[0][100:102]
        
        age = int(np.argmax(age_preds))
        gender = "Male" if np.argmax(gender_preds) == 0 else "Female"
        gender_confidence = float(np.max(gender_preds))
        
        return {
            "detections": [{
                "age": age,
                "gender": gender,
                "gender_confidence": gender_confidence
            }]
        }
    
    def _format_emotion(self, output: np.ndarray) -> Dict:
        """Format emotion recognition output"""
        emotions = ["neutral", "happy", "sad", "surprise", "anger", "disgust", "fear"]
        
        probs = output[0]
        emotion_id = np.argmax(probs)
        emotion = emotions[emotion_id] if emotion_id < len(emotions) else "unknown"
        confidence = float(probs[emotion_id])
        
        return {
            "detections": [{
                "emotion": emotion,
                "confidence": confidence
            }]
        }
    
    def _format_nsfw(self, output: np.ndarray) -> Dict:
        """Format NSFW detection output"""
        probs = output[0]
        
        safe_prob = float(probs[0])
        nsfw_prob = float(probs[1]) if len(probs) > 1 else 1.0 - safe_prob
        
        is_nsfw = nsfw_prob > self.confidence_threshold
        
        return {
            "detections": [{
                "is_nsfw": is_nsfw,
                "nsfw_probability": nsfw_prob,
                "safe_probability": safe_prob,
            }]
        }

# ============================================================================
# PERFORMANCE MONITOR
# ============================================================================

class PerformanceMonitor:
    """Monitor and log inference performance"""
    
    def __init__(self):
        self.metrics = {}
    
    def add_inference_time(self, model_name: str, elapsed_seconds: float):
        """Log inference time"""
        if model_name not in self.metrics:
            self.metrics[model_name] = []
        self.metrics[model_name].append(elapsed_seconds)
    
    def get_average_time(self, model_name: str) -> float:
        """Get average inference time for model"""
        if model_name not in self.metrics or not self.metrics[model_name]:
            return 0.0
        return np.mean(self.metrics[model_name])
    
    def get_fps(self, model_name: str) -> float:
        """Get FPS for model"""
        avg_time = self.get_average_time(model_name)
        return 1.0 / avg_time if avg_time > 0 else 0.0
    
    def get_report(self) -> Dict:
        """Get performance report"""
        report = {}
        for model_name, times in self.metrics.items():
            if times:
                report[model_name] = {
                    "avg_time_ms": np.mean(times) * 1000,
                    "min_time_ms": np.min(times) * 1000,
                    "max_time_ms": np.max(times) * 1000,
                    "fps": 1.0 / np.mean(times) if np.mean(times) > 0 else 0.0
                }
        return report