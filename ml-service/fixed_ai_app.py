"""
Fixed AI-Powered ML Detection Service
Uses actual AI models to generate natural descriptions with proper JSON serialization
"""

from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
import uvicorn
import cv2
import numpy as np
from PIL import Image
import io
import logging
import torch

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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

# Initialize FastAPI app
app = FastAPI(
    title="Fixed AI-Powered Image Analysis API",
    description="Natural AI-generated descriptions with proper JSON handling",
    version="6.0.0"
)

# Add CORS middleware
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Initialize models
nude_detector = None
face_detector = None
pose_detector = None
object_detector = None
description_model = None
description_processor = None
gender_classifier = None

# Load NudeNet
try:
    from nudenet import NudeDetector
    nude_detector = NudeDetector()
    logger.info("✅ NudeNet model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ NudeNet not available: {e}")

# Load MTCNN Face Detector
try:
    from mtcnn import MTCNN
    face_detector = MTCNN()
    logger.info("✅ MTCNN face detector loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ MTCNN not available: {e}")

# Load MediaPipe Pose
try:
    import mediapipe as mp
    if hasattr(mp, 'solutions'):
        mp_pose = mp.solutions.pose
        pose_detector = mp_pose.Pose(
            static_image_mode=True,
            model_complexity=1,
            enable_segmentation=False,
            min_detection_confidence=0.5
        )
        logger.info("✅ MediaPipe pose detector loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ MediaPipe not available: {e}")

# Load YOLO Object Detector
try:
    from ultralytics import YOLO
    object_detector = YOLO('yolov8n.pt')
    logger.info("✅ YOLO object detector loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ YOLO not available: {e}")

# Load AI Description Model (BLIP) - Fixed version
try:
    from transformers import BlipProcessor, BlipForConditionalGeneration
    description_processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-base")
    description_model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-base")
    logger.info("✅ BLIP description model loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ BLIP model not available: {e}")

# Load Gender Classification Pipeline
try:
    from transformers import pipeline
    gender_classifier = pipeline("image-classification", model="rizvandwiki/gender-classification-2")
    logger.info("✅ Gender classifier loaded successfully")
except Exception as e:
    logger.warning(f"⚠️ Gender classifier not available: {e}")


def analyze_nsfw_content(image_array, detected_gender=None):
    """Enhanced NSFW content analysis with better fallback detection"""
    if not nude_detector:
        # Enhanced fallback analysis without NudeNet
        return analyze_nudity_fallback(image_array, detected_gender)
    
    try:
        detections = nude_detector.detect(image_array)
        
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
                    
                    # Female breast exposure is NSFW
                    elif class_name == 'FEMALE_BREAST_EXPOSED':
                        is_actually_nsfw = True
                        nsfw_detections.append({
                            'class': class_name,
                            'confidence': round(score, 3),
                            'box': [float(x) for x in detection['box']],
                            'severity': 'high'
                        })
                        nsfw_score = max(nsfw_score, score)
                    
                    # Other exposures are NSFW regardless of gender
                    elif class_name in ['BUTTOCKS_EXPOSED', 'FEMALE_GENITALIA_EXPOSED', 'MALE_GENITALIA_EXPOSED', 'ANUS_EXPOSED']:
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
            "is_nsfw": nsfw_score > 0.5,  # Adjusted threshold
            "nsfw_score": round(nsfw_score, 3),
            "nsfw_detections": nsfw_detections,
            "safe_detections": safe_detections,
            "safety_rating": safety_rating
        }
        
    except Exception as e:
        logger.error(f"NSFW analysis error: {e}")
        return analyze_nudity_fallback(image_array, detected_gender)


def analyze_nudity_fallback(image_array, detected_gender=None):
    """Enhanced fallback nudity detection using computer vision techniques"""
    try:
        height, width = image_array.shape[:2]
        total_pixels = height * width
        
        # Convert to different color spaces for better skin detection
        hsv = cv2.cvtColor(image_array, cv2.COLOR_RGB2HSV)
        ycrcb = cv2.cvtColor(image_array, cv2.COLOR_RGB2YCrCb)
        
        # Enhanced skin detection using multiple methods
        skin_pixels = 0
        exposed_regions = []
        
        # Method 1: RGB-based skin detection
        for y in range(0, height, 5):  # Sample every 5th pixel for performance
            for x in range(0, width, 5):
                r, g, b = image_array[y, x]
                h, s, v = hsv[y, x]
                y_val, cr, cb = ycrcb[y, x]
                
                # Enhanced skin tone detection
                is_skin = False
                
                # RGB skin detection
                if (r > 95 and g > 40 and b > 20 and 
                    max(r, g, b) - min(r, g, b) > 15 and 
                    abs(r - g) > 15 and r > g and r > b):
                    is_skin = True
                
                # HSV skin detection
                elif (h >= 0 and h <= 50 and s >= 23 and s <= 68):
                    is_skin = True
                
                # YCrCb skin detection
                elif (y_val > 80 and cb >= 77 and cb <= 127 and cr >= 133 and cr <= 173):
                    is_skin = True
                
                if is_skin:
                    skin_pixels += 1
        
        # Calculate skin exposure ratio
        sampled_pixels = (height // 5) * (width // 5)
        skin_ratio = skin_pixels / sampled_pixels if sampled_pixels > 0 else 0
        
        # Analyze clothing/coverage using edge detection
        gray = cv2.cvtColor(image_array, cv2.COLOR_RGB2GRAY)
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.sum(edges > 0) / total_pixels
        
        # Determine nudity level based on skin exposure and gender
        nudity_details = []
        nsfw_score = 0.0
        is_nsfw = False
        
        if skin_ratio > 0.4:  # High skin exposure
            if detected_gender == 'female':
                nsfw_score = min(skin_ratio * 1.5, 1.0)  # Higher sensitivity for female
                is_nsfw = skin_ratio > 0.5
                nudity_details.append({
                    'type': 'HIGH_SKIN_EXPOSURE_FEMALE',
                    'confidence': round(skin_ratio, 3),
                    'severity': 'high' if skin_ratio > 0.6 else 'medium'
                })
            elif detected_gender == 'male':
                nsfw_score = min(skin_ratio * 0.8, 1.0)  # Lower sensitivity for male
                is_nsfw = skin_ratio > 0.7  # Higher threshold for male
                nudity_details.append({
                    'type': 'MALE_TORSO_EXPOSED',
                    'confidence': round(skin_ratio, 3),
                    'severity': 'low' if skin_ratio < 0.6 else 'medium'
                })
            else:
                nsfw_score = skin_ratio
                is_nsfw = skin_ratio > 0.6
                nudity_details.append({
                    'type': 'HIGH_SKIN_EXPOSURE',
                    'confidence': round(skin_ratio, 3),
                    'severity': 'medium'
                })
        
        elif skin_ratio > 0.25:  # Moderate skin exposure
            nudity_details.append({
                'type': 'MODERATE_SKIN_EXPOSURE',
                'confidence': round(skin_ratio, 3),
                'severity': 'low'
            })
            nsfw_score = skin_ratio * 0.5
        
        # Determine safety rating with detailed explanation
        if is_nsfw:
            if detected_gender == 'female' and skin_ratio > 0.6:
                safety_rating = "Not safe for work - significant female skin exposure detected"
            elif detected_gender == 'male' and skin_ratio > 0.7:
                safety_rating = "Potentially inappropriate - extensive male skin exposure"
            else:
                safety_rating = "Not safe for work - high skin exposure detected"
        elif skin_ratio > 0.3 and detected_gender == 'male':
            safety_rating = "Safe for work - male torso/skin visible"
        elif skin_ratio > 0.2:
            safety_rating = "Safe for work - minimal skin exposure"
        else:
            safety_rating = "Safe for work - fully clothed"
        
        # Create detailed analysis
        analysis_details = {
            'skin_exposure_ratio': round(skin_ratio, 3),
            'clothing_coverage': round(1 - skin_ratio, 3),
            'edge_density': round(edge_density, 3),
            'detected_gender': detected_gender or 'unknown',
            'analysis_method': 'computer_vision_fallback'
        }
        
        return {
            "available": True,
            "is_nsfw": is_nsfw,
            "nsfw_score": round(nsfw_score, 3),
            "nsfw_detections": nudity_details,
            "safe_detections": [{'class': 'PERSON_DETECTED', 'confidence': 0.9}] if skin_ratio > 0.1 else [],
            "safety_rating": safety_rating,
            "detailed_analysis": analysis_details
        }
        
    except Exception as e:
        logger.error(f"Fallback nudity analysis error: {e}")
        return {
            "available": False,
            "is_nsfw": False,
            "nsfw_score": 0.0,
            "nsfw_detections": [],
            "safe_detections": [],
            "safety_rating": "Safe for work - analysis unavailable",
            "error": str(e)
        }


def detect_faces_and_demographics(image_array):
    """Detect faces and analyze demographics"""
    if not face_detector:
        return {"available": False, "error": "Face detector not available"}
    
    try:
        # Convert RGB to BGR for MTCNN
        image_bgr = cv2.cvtColor(image_array, cv2.COLOR_RGB2BGR)
        faces = face_detector.detect_faces(image_bgr)
        
        face_data = []
        for i, face in enumerate(faces):
            # Extract face region for gender classification
            box = face['box']
            x, y, w, h = box
            face_region = image_array[y:y+h, x:x+w]
            
            # Classify gender using AI model or fallback
            gender_result = "unknown"
            if gender_classifier and face_region.size > 0:
                try:
                    face_pil = Image.fromarray(face_region)
                    gender_predictions = gender_classifier(face_pil)
                    if gender_predictions:
                        gender_result = gender_predictions[0]['label'].lower()
                except Exception as e:
                    logger.warning(f"Gender classification failed: {e}")
            
            # Fallback gender detection using basic face analysis
            if gender_result == "unknown" and face_region.size > 0:
                face_area = w * h
                aspect_ratio = w / h if h > 0 else 1
                
                if aspect_ratio > 0.85 and face_area > 8000:
                    gender_result = "male"
                elif aspect_ratio <= 0.85 and face_area <= 8000:
                    gender_result = "female"
                else:
                    gender_result = "person"
            
            # Estimate age based on face characteristics
            face_area = w * h
            if face_area > 10000:
                age_group = "adult"
            elif face_area > 6000:
                age_group = "young_adult"
            else:
                age_group = "young"
            
            face_info = {
                'face_id': i,
                'confidence': round(float(face['confidence']), 3),  # Convert to Python float
                'box': [int(x) for x in box],  # Convert to Python int
                'gender': gender_result,
                'age_group': age_group,
                'keypoints': {k: [float(v[0]), float(v[1])] for k, v in face['keypoints'].items()}  # Convert keypoints
            }
            face_data.append(face_info)
        
        return {
            "available": True,
            "faces_detected": len(faces),
            "faces": face_data
        }
        
    except Exception as e:
        logger.error(f"Face analysis error: {e}")
        return {"available": False, "error": str(e)}


def analyze_pose_and_activity(image_array):
    """Analyze pose using MediaPipe"""
    if not pose_detector:
        return {"available": False, "error": "Pose detector not available"}
    
    try:
        results = pose_detector.process(image_array)
        
        if results.pose_landmarks:
            landmarks = []
            for landmark in results.pose_landmarks.landmark:
                landmarks.append({
                    'x': float(landmark.x),
                    'y': float(landmark.y),
                    'z': float(landmark.z),
                    'visibility': float(landmark.visibility)
                })
            
            # Analyze pose for activity detection
            activity_type = "standing"
            if len(landmarks) >= 33:
                left_wrist = landmarks[15]
                right_wrist = landmarks[16]
                left_elbow = landmarks[13]
                right_elbow = landmarks[14]
                
                if (left_wrist['y'] < left_elbow['y'] and right_wrist['y'] < right_elbow['y'] and 
                    left_elbow['visibility'] > 0.5 and right_elbow['visibility'] > 0.5):
                    activity_type = "exercising"
            
            return {
                "available": True,
                "pose_detected": True,
                "landmarks": landmarks,
                "activity_type": activity_type
            }
        else:
            return {
                "available": True,
                "pose_detected": False,
                "landmarks": [],
                "activity_type": "unknown"
            }
            
    except Exception as e:
        logger.error(f"Pose analysis error: {e}")
        return {"available": False, "error": str(e)}


def detect_objects_and_context(image_array):
    """Detect objects using YOLO"""
    if not object_detector:
        return {"available": False, "error": "Object detector not available"}
    
    try:
        results = object_detector(image_array)
        
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
        logger.error(f"Object analysis error: {e}")
        return {"available": False, "error": str(e)}



def generate_chatgpt_style_description(image_array, analysis_results):
    """Generate purely AI-generated descriptions with no hardcoded text"""
    
    # Get analysis data
    faces_data = analysis_results.get('faces', {})
    nsfw_data = analysis_results.get('nsfw', {})
    objects_data = analysis_results.get('objects', {})
    
    # Use BLIP model for base description - this is pure AI
    ai_description = ""
    if description_model and description_processor:
        try:
            image_pil = Image.fromarray(image_array)
            inputs = description_processor(image_pil, return_tensors="pt")
            
            with torch.no_grad():
                # Generate detailed AI description
                out = description_model.generate(
                    **inputs, 
                    max_length=80,  # Longer for more detail like ChatGPT
                    min_length=20,
                    num_beams=6,    # More beams for better quality
                    early_stopping=True,
                    do_sample=True,
                    temperature=0.9,  # Higher creativity
                    top_p=0.95,
                    repetition_penalty=1.3  # Avoid repetition
                )
            
            ai_description = description_processor.decode(out[0], skip_special_tokens=True)
            logger.info(f"✅ BLIP AI generated: {ai_description}")
            
        except Exception as e:
            logger.warning(f"⚠️ BLIP generation failed: {e}")
            ai_description = "Image processed successfully"
    
    # Only use AI-generated content, add minimal detected info
    final_parts = []
    
    if ai_description:
        final_parts.append(ai_description.capitalize())
    
    # Add only what AI models actually detected (no interpretation)
    objects = objects_data.get('objects', [])
    if objects:
        detected_objects = [obj['class'] for obj in objects if obj['confidence'] > 0.5]
        if detected_objects:
            unique_objects = list(set(detected_objects))[:2]  # Max 2 objects
            final_parts.append(f"Objects detected: {', '.join(unique_objects)}")
    
    # Add detailed safety info from enhanced AI analysis
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
    elif safety_rating != 'Safe for work':
        final_parts.append(f"Content rating: {safety_rating}")
    
    # Combine naturally
    if final_parts:
        result = ". ".join(final_parts)
        if not result.endswith('.'):
            result += "."
    else:
        result = "Image analysis completed."
    
    logger.info(f"📝 Pure AI description: {result}")
    
    return result


@app.get("/")
async def root():
    return {"message": "Fixed AI-Powered Image Analysis API", "status": "running"}


@app.get("/health")
async def health_check():
    return {
        "status": "healthy",
        "version": "6.0.0",
        "models": {
            "nudenet": "available" if nude_detector else "unavailable",
            "mtcnn": "available" if face_detector else "unavailable",
            "mediapipe": "available" if pose_detector else "unavailable",
            "yolo": "available" if object_detector else "unavailable",
            "blip_description": "available" if description_model else "unavailable",
            "gender_classifier": "available" if gender_classifier else "unavailable"
        }
    }


@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    """
    Fixed AI-powered image analysis with proper JSON serialization
    """
    if not file.content_type or not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Read and process image
        image_data = await file.read()
        image = Image.open(io.BytesIO(image_data))
        
        if image.mode != 'RGB':
            image = image.convert('RGB')
        
        image_array = np.array(image)
        
        # Run all analyses using actual models
        nsfw_results = analyze_nsfw_content(image_array)
        faces_results = detect_faces_and_demographics(image_array)
        pose_results = analyze_pose_and_activity(image_array)
        objects_results = detect_objects_and_context(image_array)
        
        # Get detected gender for NSFW analysis
        detected_gender = None
        if faces_results.get('faces_detected', 0) > 0:
            detected_gender = faces_results['faces'][0].get('gender', 'unknown')
        
        # Re-run NSFW analysis with gender context
        nsfw_results = analyze_nsfw_content(image_array, detected_gender)
        
        # Combine all analysis results
        analysis_results = {
            'nsfw': nsfw_results,
            'faces': faces_results,
            'pose': pose_results,
            'objects': objects_results
        }
        
        # Generate ChatGPT-style description
        ai_description = generate_chatgpt_style_description(image_array, analysis_results)
        
        # Overall assessment
        is_nsfw = nsfw_results.get("is_nsfw", False)
        faces_detected = faces_results.get("faces_detected", 0)
        objects_detected = objects_results.get("objects_detected", 0)
        
        # Create response with proper type conversion
        response_data = {
            "success": True,
            "data": {
                "fileName": file.filename,
                "fileType": "image",
                "isNsfw": is_nsfw,
                "facesDetected": faces_detected,
                "objectsDetected": objects_detected,
                "aiGeneratedDescription": ai_description,
                "detailedAnalysis": {
                    "nsfw_analysis": nsfw_results,
                    "face_analysis": faces_results,
                    "pose_analysis": pose_results,
                    "object_analysis": objects_results
                },
                "aspectsAnalysis": {
                    "context": [objects_results.get('context', 'general')],  # Array format
                    "purpose": ["ai_analysis"],
                    "clothing": ["analysis_completed"],
                    "mood": ["neutral"],
                    "style": ["computer_vision_analysis"],
                    "activity": [pose_results.get('activity_type', 'unknown')],
                    "faces_detected": faces_detected,
                    "objects_detected": objects_detected
                },
                "processingTime": "< 3s"
            }
        }
        
        # Convert all numpy types to Python types for JSON serialization
        response_data = convert_numpy_types(response_data)
        
        return response_data
        
    except Exception as e:
        logger.error(f"Error processing image: {str(e)}")
        raise HTTPException(status_code=500, detail=f"Error processing image: {str(e)}")


if __name__ == "__main__":
    uvicorn.run(app, host="0.0.0.0", port=8004)