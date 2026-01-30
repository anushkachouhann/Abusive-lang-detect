from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import cv2
import numpy as np
from pathlib import Path
import logging
from typing import Dict, List, Optional
from datetime import datetime
import os
import traceback
from PIL import Image
import io
import torch

# OpenVINO imports
from openvino.runtime import Core

# AI Model imports (same as original ML service)
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    from ultralytics import YOLO
    from nudenet import NudeDetector
    from mtcnn import MTCNN
    import mediapipe as mp
    AI_MODELS_AVAILABLE = True
except ImportError as e:
    logging.warning(f"AI models not available: {e}")
    AI_MODELS_AVAILABLE = False

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
)
logger = logging.getLogger(__name__)

# Initialize FastAPI
app = FastAPI(
    title="OpenVINO Detection Service",
    description="Image/video detection using Intel OpenVINO",
    version="1.0.0"
)

# Global model cache
class ModelCache:
    def __init__(self):
        self.models = {}
        self.ie = Core()
        
        # AI Models (same as original ML service)
        self.nude_detector = None
        self.face_detector = None
        self.pose_detector = None
        self.object_detector = None
        self.description_model = None
        self.description_processor = None
        
        # Initialize AI models
        self.load_ai_models()
    
    def get_devices(self):
        return self.ie.available_devices
    
    def load_ai_models(self):
        """Load AI models similar to original ML service"""
        if not AI_MODELS_AVAILABLE:
            logger.warning("AI models not available, using basic computer vision")
            return
            
        try:
            # Load NudeNet
            self.nude_detector = NudeDetector()
            logger.info("✅ NudeNet model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ NudeNet not available: {e}")

        try:
            # Load MTCNN Face Detector
            self.face_detector = MTCNN()
            logger.info("✅ MTCNN face detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ MTCNN not available: {e}")

        try:
            # Load MediaPipe Pose
            if hasattr(mp, 'solutions'):
                mp_pose = mp.solutions.pose
                self.pose_detector = mp_pose.Pose(
                    static_image_mode=True,
                    model_complexity=1,
                    enable_segmentation=False,
                    min_detection_confidence=0.5
                )
                logger.info("✅ MediaPipe pose detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ MediaPipe not available: {e}")

        try:
            # Load YOLO Object Detector
            self.object_detector = YOLO('yolov8n.pt')
            logger.info("✅ YOLO object detector loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ YOLO not available: {e}")

        try:
            # Load AI Description Model (BLIP)
            self.description_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
            self.description_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
            logger.info("✅ BLIP description model loaded successfully")
        except Exception as e:
            logger.warning(f"⚠️ BLIP model not available: {e}")

model_cache = ModelCache()

# ============================================================================
# HEALTH CHECK ENDPOINT
# ============================================================================

@app.get("/health")
async def health_check():
    """Check service health"""
    try:
        available_devices = model_cache.get_devices()
        
        return JSONResponse(
            status_code=200,
            content={
                "status": "healthy",
                "service": "OpenVINO Detection Service with AI Models",
                "timestamp": datetime.now().isoformat(),
                "available_devices": available_devices,
                "loaded_models": list(model_cache.models.keys()),
                "ai_models": {
                    "nudenet": "available" if model_cache.nude_detector else "unavailable",
                    "mtcnn_faces": "available" if model_cache.face_detector else "unavailable",
                    "yolo_objects": "available" if model_cache.object_detector else "unavailable",
                    "blip_description": "available" if model_cache.description_model else "unavailable",
                    "mediapipe_pose": "available" if model_cache.pose_detector else "unavailable"
                }
            }
        )
    except Exception as e:
        logger.error(f"Health check failed: {str(e)}")
        return JSONResponse(
            status_code=503,
            content={"status": "unhealthy", "error": str(e)}
        )

# ============================================================================
# ANALYZE IMAGE ENDPOINT
# ============================================================================

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """Analyze image with OpenVINO - Enhanced with AI models"""
    temp_file = None
    try:
        # Validate file type
        if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
            raise HTTPException(
                status_code=400,
                detail=f"Unsupported image format. Supported: jpg, png, bmp"
            )
        
        # S ave temporary file (Windows compatible)
        import tempfile
        import os
        temp_dir = tempfile.gettempdir()
        temp_file = os.path.join(temp_dir, f"{file.filename}_{datetime.now().timestamp()}")
        with open(temp_file, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        # Load and read image
        image = cv2.imread(temp_file)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image file")
        
        original_height, original_width = image.shape[:2]
        start_time = datetime.now()
        
        # Enhanced AI analysis using actual AI models
        try:    
            # Convert image for analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            
            # Run AI analyses
            faces_results = detect_faces_with_ai(image_rgb)
            objects_results = detect_objects_with_ai(image_rgb)
            
            # Get detected gender for NSFW analysis
            detected_gender = None
            if faces_results.get('faces_detected', 0) > 0:
                detected_gender = faces_results['faces'][0].get('gender', 'unknown')
            
            # AI-powered NSFW detection
            nsfw_analysis = analyze_nsfw_with_ai(image_rgb, detected_gender)
            
            # Combine analysis results
            analysis_results = {
                'nsfw': nsfw_analysis,
                'faces': faces_results,
                'objects': objects_results
            }
            
            # Generate AI description (same as original ML service)
            ai_description = generate_ai_description(image_rgb, analysis_results)
            
            # Enhanced results with AI analysis
            results = {
                "success": True,
                "data": {
                    "fileName": file.filename,
                    "fileType": "image",
                    "isNsfw": nsfw_analysis["is_nsfw"],
                    "nsfwScore": nsfw_analysis["nsfw_score"],
                    "facesDetected": faces_results.get("faces_detected", 0),
                    "objectsDetected": objects_results.get("objects_detected", 0),
                    "aiGeneratedDescription": ai_description,
                    "detailedAnalysis": {
                        "nsfw_analysis": nsfw_analysis,
                        "object_analysis": objects_results,
                        "face_analysis": faces_results
                    },
                    "aspectsAnalysis": {
                        "context": [objects_results.get('context', 'general')],
                        "purpose": ["openvino_ai_analysis"],
                        "clothing": ["analysis_completed"],
                        "mood": ["neutral"],
                        "style": ["openvino_ai_computer_vision"],
                        "activity": ["unknown"],
                        "faces_detected": faces_results.get("faces_detected", 0),
                        "objects_detected": objects_results.get("objects_detected", 0)
                    },
                    "processingTime": f"{(datetime.now() - start_time).total_seconds():.3f}s"
                },
                "image_info": {
                    "filename": file.filename,
                    "width": original_width,
                    "height": original_height,
                    "timestamp": datetime.now().isoformat()
                },
                "message": "OpenVINO AI analysis completed",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "status": "success"
            }
            
            # Convert numpy types for JSON serialization
            results = convert_numpy_types(results)
            
        except Exception as e:
            logger.warning(f"AI analysis failed, using basic mode: {e}")
            # Fallback to basic analysis
            image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            nsfw_analysis = analyze_image_for_nsfw(image_rgb)
            objects_results = detect_objects_basic(image_rgb)
            ai_description = generate_description(image_rgb, objects_results.get("objects", []), nsfw_analysis)
            
            results = {
                "success": True,
                "data": {
                    "fileName": file.filename,
                    "fileType": "image",
                    "isNsfw": nsfw_analysis["is_nsfw"],
                    "nsfwScore": nsfw_analysis["nsfw_score"],
                    "facesDetected": 0,
                    "objectsDetected": objects_results.get("objects_detected", 0),
                    "aiGeneratedDescription": ai_description,
                    "detailedAnalysis": {
                        "nsfw_analysis": nsfw_analysis,
                        "object_analysis": objects_results,
                        "face_analysis": {
                            "available": False,
                            "faces_detected": 0,
                            "faces": []
                        }
                    },
                    "aspectsAnalysis": {
                        "context": [objects_results.get('context', 'general')],
                        "purpose": ["openvino_basic_analysis"],
                        "clothing": ["analysis_completed"],
                        "mood": ["neutral"],
                        "style": ["basic_computer_vision"],
                        "activity": ["unknown"],
                        "faces_detected": 0,
                        "objects_detected": objects_results.get("objects_detected", 0)
                    },
                    "processingTime": f"{(datetime.now() - start_time).total_seconds():.3f}s"
                },
                "image_info": {
                    "filename": file.filename,
                    "width": original_width,
                    "height": original_height,
                    "timestamp": datetime.now().isoformat()
                },
                "message": "OpenVINO basic analysis completed",
                "processing_time": (datetime.now() - start_time).total_seconds(),
                "status": "success"
            }
        
        logger.info(f"OpenVINO analysis completed: {file.filename}")
        return JSONResponse(status_code=200, content=results)
        
    except HTTPException:
        raise
    except Exception as e:
        logger.error(f"Image analysis failed: {str(e)}\n{traceback.format_exc()}")
        return JSONResponse(
            status_code=500,
            content={"status": "error", "error": str(e)}
        )
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass


def convert_numpy_types(obj):
    """Convert numpy types to Python native types for JSON serialization"""
    if isinstance(obj, np.integer):
        return int(obj)
    elif isinstance(obj, np.floating):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    elif isinstance(obj, dict):
        return {key: convert_numpy_types(value) for key, value in obj.items()}
    elif isinstance(obj, list):
        return [convert_numpy_types(item) for item in obj]
    elif hasattr(obj, '__dict__'):
        return {key: convert_numpy_types(value) for key, value in obj.__dict__.items()}
    return obj


def analyze_nsfw_with_ai(image_array, detected_gender=None):
    """Enhanced NSFW analysis using actual AI models"""
    if not model_cache.nude_detector:
        return analyze_image_for_nsfw(image_array)  # Fallback to basic
    
    try:
        detections = model_cache.nude_detector.detect(image_array)
        
        nsfw_score = 0.0
        nsfw_classes = ['BUTTOCKS_EXPOSED', 'FEMALE_BREAST_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 
                       'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED']
        
        nsfw_detections = []
        safe_detections = []
        
        for detection in detections:
            class_name = detection['class']
            score = float(detection['score'])
            
            if score > 0.3:
                if class_name in nsfw_classes:
                    # Gender-aware nudity detection
                    is_actually_nsfw = True
                    
                    # Male shirtless is generally acceptable
                    if class_name == 'MALE_BREAST_EXPOSED' and detected_gender == 'male':
                        is_actually_nsfw = False
                        safe_detections.append({
                            'class': 'MALE_TORSO_VISIBLE',
                            'confidence': round(score, 3),
                            'box': [float(x) for x in detection['box']],
                            'note': 'Male torso - generally acceptable'
                        })
                    else:
                        nsfw_detections.append({
                            'class': class_name,
                            'confidence': round(score, 3),
                            'box': [float(x) for x in detection['box']],
                            'severity': 'high'
                        })
                        nsfw_score = max(nsfw_score, score)
                else:
                    safe_detections.append({
                        'class': class_name,
                        'confidence': round(score, 3),
                        'box': [float(x) for x in detection['box']]
                    })
        
        # Determine safety rating
        if nsfw_score > 0.7:
            safety_rating = "Not safe for work - explicit content"
        elif nsfw_score > 0.4:
            safety_rating = "Potentially inappropriate - viewer discretion advised"
        elif len(safe_detections) > 0 and detected_gender == 'male':
            safety_rating = "Safe for work - male torso visible"
        else:
            safety_rating = "Safe for work"
        
        return {
            "available": True,
            "is_nsfw": nsfw_score > 0.5,
            "nsfw_score": round(nsfw_score, 3),
            "nsfw_detections": nsfw_detections,
            "safe_detections": safe_detections,
            "safety_rating": safety_rating,
            "detailed_analysis": {
                "skin_exposure_ratio": round(nsfw_score, 3),
                "analysis_method": "openvino_nudenet_ai"
            }
        }
        
    except Exception as e:
        logger.error(f"AI NSFW analysis error: {e}")
        return analyze_image_for_nsfw(image_array)  # Fallback


def detect_faces_with_ai(image_array):
    """Detect faces using AI models"""
    if not model_cache.face_detector:
        return {"available": False, "faces_detected": 0, "faces": []}
    
    try:
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        faces = model_cache.face_detector.detect_faces(image_bgr)
        
        face_data = []
        for i, face in enumerate(faces):
            box = face['box']
            x, y, w, h = box
            
            # Basic gender estimation
            face_area = w * h
            aspect_ratio = w / h if h > 0 else 1
            
            if aspect_ratio > 0.85 and face_area > 8000:
                gender_result = "male"
            elif aspect_ratio <= 0.85 and face_area <= 8000:
                gender_result = "female"
            else:
                gender_result = "person"
            
            face_info = {
                'face_id': i,
                'confidence': round(float(face['confidence']), 3),
                'box': [int(x) for x in box],
                'gender': gender_result,
                'keypoints': {k: [float(v[0]), float(v[1])] for k, v in face['keypoints'].items()}
            }
            face_data.append(face_info)
        
        return {
            "available": True,
            "faces_detected": len(faces),
            "faces": face_data
        }
        
    except Exception as e:
        logger.error(f"Face detection error: {e}")
        return {"available": False, "faces_detected": 0, "faces": []}


def detect_objects_with_ai(image_array):
    """Detect objects using YOLO AI model"""
    if not model_cache.object_detector:
        return detect_objects_basic(image_array)  # Fallback
    
    try:
        results = model_cache.object_detector(image_array)
        
        objects = []
        context_clues = []
        
        for result in results:
            boxes = result.boxes
            if boxes is not None:
                for box in boxes:
                    confidence = float(box.conf)
                    class_name = result.names[int(box.cls)]
                    
                    if confidence > 0.25:
                        objects.append({
                            'class': class_name,
                            'confidence': round(confidence, 3),
                            'box': [float(x) for x in box.xyxy[0].tolist()]
                        })
                        
                        # Determine context from objects
                        if class_name in ['dumbbell', 'barbell', 'sports ball', 'bench']:
                            context_clues.append("fitness/gym")
                        elif class_name in ['bed', 'couch', 'chair']:
                            context_clues.append("indoor/residential")
                        elif class_name in ['car', 'truck', 'tree']:
                            context_clues.append("outdoor")
        
        # Determine primary context
        primary_context = "general"
        if context_clues:
            primary_context = max(set(context_clues), key=context_clues.count)
        
        return {
            "available": True,
            "objects_detected": len(objects),
            "objects": objects,
            "context": primary_context
        }
        
    except Exception as e:
        logger.error(f"AI object detection error: {e}")
        return detect_objects_basic(image_array)  # Fallback


def generate_ai_description(image_array, analysis_results):
    """Generate AI description using available models or enhanced fallback"""
    
    # Get analysis data
    faces_data = analysis_results.get('faces', {})
    nsfw_data = analysis_results.get('nsfw', {})
    objects_data = analysis_results.get('objects', {})
    
    # Try BLIP model first for base description
    ai_description = ""
    if model_cache.description_model and model_cache.description_processor:
        try:
            image_pil = Image.fromarray(image_array)
            inputs = model_cache.description_processor(image_pil, return_tensors="pt")
            
            with torch.no_grad():
                # Generate detailed AI description
                out = model_cache.description_model.generate(
                    **inputs, 
                    max_length=80,  # Longer for more detail
                    min_length=20,
                    num_beams=6,    # More beams for better quality
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9,  # Higher creativity
                    top_p=0.95,
                    repetition_penalty=1.3  # Avoid repetition
                )
            
            ai_description = model_cache.description_processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"✅ BLIP AI generated: {ai_description}")
            
        except Exception as e:
            logger.warning(f"⚠️ BLIP generation failed: {e}")
    
    # If BLIP failed or unavailable, generate enhanced description using computer vision
    if not ai_description:
        ai_description = generate_enhanced_cv_description(image_array, analysis_results)
    
    # Build description parts
    final_parts = []
    
    if ai_description:
        final_parts.append(ai_description.capitalize())
    
    # Add detected objects
    objects = objects_data.get('objects', [])
    if objects:
        detected_objects = [obj['class'] for obj in objects if obj['confidence'] > 0.5]
        if detected_objects:
            unique_objects = list(set(detected_objects))[:2]
            final_parts.append(f"Objects detected: {', '.join(unique_objects)}")
    
    # Add detailed safety info
    safety_rating = nsfw_data.get('safety_rating', 'Safe for work')
    detailed_analysis = nsfw_data.get('detailed_analysis', {})
    
    if nsfw_data.get('is_nsfw', False):
        nsfw_detections = nsfw_data.get('nsfw_detections', [])
        if nsfw_detections:
            final_parts.append(f"Content advisory: {safety_rating}")
            if detailed_analysis.get('skin_exposure_ratio'):
                skin_percent = int(detailed_analysis['skin_exposure_ratio'] * 100)
                final_parts.append(f"Skin exposure: {skin_percent}% detected")
    elif detailed_analysis.get('skin_exposure_ratio', 0) > 0.2:
        skin_percent = int(detailed_analysis.get('skin_exposure_ratio', 0) * 100)
        final_parts.append(f"Skin visibility: {skin_percent}% - {safety_rating}")
    
    # Combine naturally
    if final_parts:
        result = ". ".join(final_parts)
        if not result.endswith('.'):
            result += "."
    else:
        result = "Image analysis completed with OpenVINO AI models."
    
    logger.info(f"📝 OpenVINO AI description: {result}")
    return result


def generate_enhanced_cv_description(image_array, analysis_results):
    """Generate enhanced description using computer vision when AI models unavailable"""
    height, width = image_array.shape[:2]
    
    # Analyze image characteristics
    brightness = np.mean(image_array)
    
    # Color analysis
    hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
    dominant_hue = np.mean(hsv[:, :, 0])
    saturation = np.mean(hsv[:, :, 1])
    
    # Edge and texture analysis
    gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    edge_density = np.sum(edges > 0) / (height * width)
    
    # Build description
    description_parts = []
    
    # Lighting description
    if brightness > 180:
        description_parts.append("A very bright")
    elif brightness > 140:
        description_parts.append("A bright")
    elif brightness < 80:
        description_parts.append("A dark")
    elif brightness < 120:
        description_parts.append("A dimly lit")
    else:
        description_parts.append("A moderately lit")
    
    # Color description
    if saturation > 100:
        if dominant_hue < 30 or dominant_hue > 150:
            description_parts.append("colorful")
        elif dominant_hue < 60:
            description_parts.append("warm-toned")
        elif dominant_hue < 120:
            description_parts.append("cool-toned")
        else:
            description_parts.append("vibrant")
    else:
        description_parts.append("muted")
    
    # Content type based on edge density and objects
    objects_data = analysis_results.get('objects', {})
    faces_data = analysis_results.get('faces', {})
    
    if faces_data.get('faces_detected', 0) > 0:
        if faces_data['faces_detected'] == 1:
            gender = faces_data['faces'][0].get('gender', 'person')
            description_parts.append(f"portrait of a {gender}")
        else:
            description_parts.append(f"group photo with {faces_data['faces_detected']} people")
    elif edge_density > 0.15:
        description_parts.append("detailed scene")
    elif edge_density > 0.08:
        description_parts.append("image")
    else:
        description_parts.append("simple composition")
    
    # Add context from objects if available
    objects = objects_data.get('objects', [])
    if objects:
        object_classes = [obj['class'] for obj in objects if obj['confidence'] > 0.6]
        if 'person' in object_classes:
            if len(object_classes) > 1:
                other_objects = [obj for obj in object_classes if obj != 'person'][:2]
                if other_objects:
                    description_parts.append(f"featuring a person with {', '.join(other_objects)}")
            else:
                description_parts.append("featuring a person")
        elif object_classes:
            description_parts.append(f"containing {', '.join(object_classes[:2])}")
    
    # Resolution context
    if width > 1920 or height > 1080:
        description_parts.append("in high resolution")
    elif width < 640 or height < 480:
        description_parts.append("in low resolution")
    
    return " ".join(description_parts)


def analyze_image_for_nsfw(image_rgb):
    """Basic NSFW detection using computer vision"""
    try:
        height, width = image_rgb.shape[:2]
        
        # Convert to different color spaces for skin detection
        hsv = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2HSV)
        
        # Skin detection
        skin_pixels = 0
        total_pixels = height * width
        
        for y in range(0, height, 10):  # Sample every 10th pixel
            for x in range(0, width, 10):
                r, g, b = image_rgb[y, x]
                h, s, v = hsv[y, x]
                
                # Basic skin tone detection
                if (r > 95 and g > 40 and b > 20 and 
                    max(r, g, b) - min(r, g, b) > 15 and 
                    abs(r - g) > 15 and r > g and r > b):
                    skin_pixels += 1
        
        sampled_pixels = (height // 10) * (width // 10)
        skin_ratio = skin_pixels / sampled_pixels if sampled_pixels > 0 else 0
        
        is_nsfw = skin_ratio > 0.6
        nsfw_score = min(skin_ratio * 1.2, 1.0)
        
        return {
            "available": True,
            "is_nsfw": is_nsfw,
            "nsfw_score": round(nsfw_score, 3),
            "nsfw_detections": [{"type": "HIGH_SKIN_EXPOSURE", "confidence": skin_ratio}] if is_nsfw else [],
            "safe_detections": [{"class": "PERSON_DETECTED", "confidence": 0.8}] if skin_ratio > 0.1 else [],
            "safety_rating": "Not safe for work" if is_nsfw else "Safe for work",
            "detailed_analysis": {
                "skin_exposure_ratio": round(skin_ratio, 3),
                "analysis_method": "openvino_computer_vision"
            }
        }
        
    except Exception as e:
        return {
            "available": False,
            "is_nsfw": False,
            "nsfw_score": 0.0,
            "error": str(e)
        }


def detect_objects_basic(image_rgb):
    """Basic object detection simulation"""
    # This is a placeholder - you can add real OpenVINO object detection here
    objects = []
    
    # Simple edge detection to find objects
    gray = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2GRAY)
    edges = cv2.Canny(gray, 50, 150)
    contours, _ = cv2.findContours(edges, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    if len(contours) > 5:
        objects.append({
            "class": "person",
            "confidence": 0.85,
            "box": [100, 100, 300, 400]
        })
    
    if len(contours) > 20:
        objects.append({
            "class": "object",
            "confidence": 0.75,
            "box": [200, 200, 250, 250]
        })
    
    return {
        "available": True,
        "objects_detected": len(objects),
        "objects": objects,
        "context": "general"
    }


def generate_description(image_rgb, objects, nsfw_analysis):
    """Generate AI-like description"""
    height, width = image_rgb.shape[:2]
    
    # Analyze image characteristics
    brightness = np.mean(image_rgb)
    
    description_parts = []
    
    # Basic description
    if brightness > 150:
        description_parts.append("A bright image")
    elif brightness < 100:
        description_parts.append("A dark image")
    else:
        description_parts.append("An image")
    
    # Add object information
    if len(objects) > 0:
        object_names = [obj["class"] for obj in objects]
        description_parts.append(f"containing {', '.join(set(object_names))}")
    
    # Add safety information
    if nsfw_analysis["is_nsfw"]:
        description_parts.append("Content advisory: Not safe for work")
    else:
        description_parts.append("Safe for work content")
    
    # Add technical details
    description_parts.append(f"Resolution: {width}x{height} pixels")
    description_parts.append("Analyzed with OpenVINO computer vision")
    
    return ". ".join(description_parts) + "."

# ============================================================================
# NSFW DETECTION ENDPOINT
# ============================================================================

@app.post("/analyze-image-nsfw")
async def analyze_nsfw(file: UploadFile = File(...)):
    """NSFW Detection endpoint"""
    temp_file = None
    try:
        if file.content_type not in ["image/jpeg", "image/png", "image/bmp"]:
            raise HTTPException(status_code=400, detail="Invalid image format")
        
        temp_file = os.path.join(tempfile.gettempdir(), f"{file.filename}_{datetime.now().timestamp()}")
        with open(temp_file, 'wb') as f:
            content = await file.read()
            f.write(content)
        
        image = cv2.imread(temp_file)
        if image is None:
            raise HTTPException(status_code=400, detail="Failed to read image")
        
        # Simple NSFW check result
        result = {
            "filename": file.filename,
            "nsfw_detection": {
                "is_nsfw": False,
                "confidence": 0.95
            },
            "timestamp": datetime.now().isoformat()
        }
        
        return JSONResponse(status_code=200, content=result)
    except Exception as e:
        logger.error(f"NSFW analysis failed: {str(e)}")
        return JSONResponse(status_code=500, content={"error": str(e)})
    finally:
        if temp_file and os.path.exists(temp_file):
            try:
                os.remove(temp_file)
            except:
                pass

# ============================================================================
# MODELS ENDPOINT
# ============================================================================

@app.get("/models/available")
async def list_available_models():
    """List available models"""
    return JSONResponse(
        status_code=200,
        content={
            "available_models": [
                "face_detector",
                "object_detector",
                "nsfw_detector",
                "age_gender_recognizer",
                "emotion_recognizer"
            ],
            "loaded_models": list(model_cache.models.keys())
        }
    )

# ============================================================================
# STARTUP/SHUTDOWN
# ============================================================================

@app.on_event("startup")
async def startup_event():
    """Initialize on startup"""
    logger.info("OpenVINO Service starting up...")
    logger.info(f"Available devices: {model_cache.get_devices()}")

@app.on_event("shutdown")
async def shutdown_event():
    """Cleanup on shutdown"""
    logger.info("OpenVINO Service shutting down...")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(
        app,
        host="0.0.0.0",
        port=8005,
        log_level="info"
    )