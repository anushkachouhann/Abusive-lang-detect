from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.responses import JSONResponse
import tempfile
import os
from nudenet import NudeDetector
import cv2
from PIL import Image
import io

app = FastAPI()
detector = NudeDetector()

def analyze_body_coverage(detections):
    """Analyze body coverage and clothing from NudeNet detections with comprehensive AI-level analysis"""
    
    # Classification mappings
    exposed_parts = {
        'EXPOSED_ANUS': 'anus',
        'EXPOSED_ARMPITS': 'armpits', 
        'EXPOSED_BELLY': 'belly',
        'BELLY_EXPOSED': 'belly',  # Alternative naming
        'EXPOSED_BUTTOCKS': 'buttocks',
        'EXPOSED_BREAST_F': 'female breast',
        'FEMALE_BREAST_EXPOSED': 'female breast',  # Alternative naming
        'EXPOSED_BREAST_M': 'male breast',
        'MALE_BREAST_EXPOSED': 'male breast',  # Alternative naming
        'EXPOSED_GENITALIA_F': 'female genitalia',
        'EXPOSED_GENITALIA_M': 'male genitalia'
    }
    
    covered_parts = {
        'COVERED_BELLY': 'belly',
        'COVERED_BUTTOCKS': 'buttocks',
        'COVERED_BREAST_F': 'female breast',
        'COVERED_BREAST_M': 'male breast',
        'COVERED_GENITALIA_F': 'female genitalia',
        'COVERED_GENITALIA_M': 'male genitalia'
    }
    
    face_parts = {
        'FACE_F': 'female',
        'FACE_M': 'male'
    }
    
    body_parts = {
        'ARMPITS_COVERED': 'covered armpits',
        'ARMPITS_EXPOSED': 'exposed armpits',
        'FEET_EXPOSED': 'exposed feet',
        'FEET_COVERED': 'covered feet',
        'HAND_COVERED': 'covered hands',
        'HAND_EXPOSED': 'exposed hands'
    }
    
    # Analyze detections
    detected_exposed = []
    detected_covered = []
    detected_faces = []
    detected_body = []
    gender_detections = []
    
    nsfw_score = 0.0
    high_confidence_detections = []
    max_confidence = 0.0
    
    for detection in detections:
        class_name = detection['class']
        confidence = detection['score']
        max_confidence = max(max_confidence, confidence)
        
        if confidence > 0.4:  # Lower threshold to catch more detections
            high_confidence_detections.append(detection)
            
            if class_name in exposed_parts:
                # Handle potential gender misclassification
                part_name = exposed_parts[class_name]
                if class_name == 'FEMALE_BREAST_EXPOSED' and confidence < 0.5:
                    # Likely misclassified male chest
                    part_name = 'male chest'
                    gender_detections.append(('male', confidence))
                elif 'FEMALE' in class_name or '_F' in class_name:
                    gender_detections.append(('female', confidence))
                elif 'MALE' in class_name or '_M' in class_name:
                    gender_detections.append(('male', confidence))
                    
                detected_exposed.append(part_name)
                nsfw_score = max(nsfw_score, confidence)
            elif class_name in covered_parts:
                part_name = covered_parts[class_name]
                if 'FEMALE' in class_name or '_F' in class_name:
                    gender_detections.append(('female', confidence))
                elif 'MALE' in class_name or '_M' in class_name:
                    gender_detections.append(('male', confidence))
                detected_covered.append(part_name)
            elif class_name in face_parts:
                gender = face_parts[class_name]
                detected_faces.append(f"{gender} face")
                gender_detections.append((gender, confidence))
            elif class_name in body_parts:
                detected_body.append(body_parts[class_name])
    
    # Determine primary gender
    detected_gender = "Unknown"
    if gender_detections:
        # Count gender occurrences weighted by confidence
        male_score = sum(conf for gender, conf in gender_detections if gender == 'male')
        female_score = sum(conf for gender, conf in gender_detections if gender == 'female')
        
        if male_score > female_score:
            detected_gender = "Male"
        elif female_score > male_score:
            detected_gender = "Female"
        else:
            detected_gender = "Mixed/Uncertain"
    
    # Calculate coverage percentage
    total_body_parts = len(detected_exposed) + len(detected_covered)
    if total_body_parts > 0:
        coverage_percentage = round((len(detected_covered) / total_body_parts) * 100, 1)
    else:
        # If no body parts detected, check if we have any detections at all
        if len(high_confidence_detections) == 0:
            coverage_percentage = 100.0  # No human detected, assume safe
        else:
            # We have detections but no body parts - might be face only or unclear image
            coverage_percentage = 50.0  # Uncertain, assume moderate coverage
    
    # Generate comprehensive professional description
    description_parts = []
    detailed_analysis_parts = []
    
    # Subject identification
    if detected_faces:
        subject_desc = f"Analysis identifies a {detected_gender.lower()} subject"
        description_parts.append(subject_desc)
        detailed_analysis_parts.append(f"SUBJECT IDENTIFICATION: {subject_desc} with facial features clearly visible.")
    elif detected_gender != "Unknown":
        subject_desc = f"Analysis suggests a {detected_gender.lower()} subject"
        description_parts.append(subject_desc)
        detailed_analysis_parts.append(f"SUBJECT IDENTIFICATION: {subject_desc} based on body characteristics.")
    
    # Coverage analysis
    if detected_covered:
        covered_desc = f"Clothed areas include: {', '.join(set(detected_covered))}"
        description_parts.append(covered_desc)
        detailed_analysis_parts.append(f"COVERED REGIONS: {covered_desc}. These areas show appropriate clothing coverage.")
    
    if detected_exposed:
        exposed_desc = f"Exposed areas include: {', '.join(set(detected_exposed))}"
        description_parts.append(exposed_desc)
        detailed_analysis_parts.append(f"EXPOSED REGIONS: {exposed_desc}. Analysis indicates skin visibility in these regions.")
    
    if detected_body:
        body_desc = f"Additional visible features: {', '.join(set(detected_body))}"
        description_parts.append(body_desc)
        detailed_analysis_parts.append(f"ADDITIONAL FEATURES: {body_desc}.")
    
    # Professional clothing assessment
    clothing_status = "Unknown"
    clothing_detail = ""
    
    if total_body_parts == 0:
        if len(detected_faces) > 0:
            clothing_status = "Portrait/Face Only"
            clothing_detail = "Image primarily shows facial features with limited body visibility for clothing assessment."
        else:
            clothing_status = "No Clear Human Subject"
            clothing_detail = "No distinct human figure detected for clothing analysis."
    elif coverage_percentage >= 90:
        clothing_status = "Fully Clothed"
        clothing_detail = "Subject appears fully dressed with comprehensive clothing coverage."
    elif coverage_percentage >= 70:
        clothing_status = "Mostly Clothed"
        clothing_detail = "Subject shows good clothing coverage with minor exposed areas."
    elif coverage_percentage >= 50:
        clothing_status = "Partially Clothed"
        clothing_detail = "Subject displays mixed clothing coverage with notable exposed regions."
    elif coverage_percentage >= 20:
        clothing_status = "Minimally Clothed"
        clothing_detail = "Subject shows limited clothing with significant skin exposure."
    else:
        clothing_status = "Unclothed/Nude"
        clothing_detail = "Subject appears without clothing or with minimal coverage."
    
    detailed_analysis_parts.append(f"CLOTHING ASSESSMENT: {clothing_status}. {clothing_detail}")
    
    # Body coverage professional assessment
    body_coverage = "Appropriate"
    coverage_detail = ""
    
    if detected_exposed:
        sensitive_areas = ['female breast', 'male breast', 'female genitalia', 'male genitalia', 'buttocks', 'anus']
        highly_sensitive = ['female breast', 'female genitalia', 'male genitalia', 'buttocks', 'anus']
        
        if any(area in detected_exposed for area in highly_sensitive):
            body_coverage = "Inappropriate Content"
            coverage_detail = "Image contains exposure of intimate/private body regions requiring content restriction."
        elif any(area in detected_exposed for area in sensitive_areas):
            body_coverage = "Moderate Exposure"
            coverage_detail = "Image shows partial nudity with exposed torso/chest areas that may require age-appropriate viewing."
        else:
            body_coverage = "Minor Exposure"
            coverage_detail = "Image contains minimal skin exposure of non-intimate areas."
    elif total_body_parts == 0 and len(high_confidence_detections) > 0:
        body_coverage = "Requires Review"
        coverage_detail = "Image analysis inconclusive, manual review recommended for accurate assessment."
    else:
        coverage_detail = "Image shows appropriate content with no concerning exposure detected."
    
    detailed_analysis_parts.append(f"CONTENT ASSESSMENT: {body_coverage}. {coverage_detail}")
    
    # Confidence and technical details
    detailed_analysis_parts.append(f"TECHNICAL DETAILS: Analysis processed {len(detections)} detections with maximum confidence of {max_confidence:.3f}. Coverage calculation: {coverage_percentage}% clothed based on {total_body_parts} identified body regions.")
    
    # Final description
    if not description_parts:
        description = "No clear human subject identified in the image for detailed analysis."
        detailed_analysis_parts = ["ANALYSIS RESULT: No distinct human figure detected. Image may contain non-human subjects, abstract content, or unclear imagery."]
    else:
        description = ". ".join(description_parts) + "."
    
    detailed_analysis = " | ".join(detailed_analysis_parts)
    
    # NSFW determination with more nuanced logic
    is_nsfw = (nsfw_score > 0.6 or 
               any(area in detected_exposed for area in ['female breast', 'female genitalia', 'male genitalia', 'buttocks', 'anus']) or
               (coverage_percentage < 30 and detected_gender == "Female"))
    
    return {
        "is_nsfw": is_nsfw,
        "nsfw_score": round(nsfw_score, 3),
        "body_coverage": body_coverage,
        "clothing_analysis": clothing_status,
        "description": description,
        "detailed_analysis": detailed_analysis,
        "detected_gender": detected_gender,
        "confidence_score": round(max_confidence, 3),
        "detected_parts": {
            "exposed": list(set(detected_exposed)),
            "covered": list(set(detected_covered)),
            "faces": list(set(detected_faces)),
            "other": list(set(detected_body))
        },
        "coverage_percentage": coverage_percentage
    }

@app.post("/debug-analyze")
async def debug_analyze(file: UploadFile = File(...)):
    """Debug endpoint to see raw detection results"""
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Analyze image
        results = detector.detect(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        # Return raw results for debugging
        return {
            "raw_detections": results,
            "detection_count": len(results),
            "high_confidence_detections": [d for d in results if d['score'] > 0.5]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.get("/health")
async def health_check():
    return {"status": "healthy"}

@app.post("/analyze-image")
async def analyze_image(file: UploadFile = File(...)):
    if not file.content_type.startswith('image/'):
        raise HTTPException(status_code=400, detail="File must be an image")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Analyze image
        results = detector.detect(tmp_file_path)
        
        # Clean up
        os.unlink(tmp_file_path)
        
        # Analyze detections and create detailed description
        analysis = analyze_body_coverage(results)
        
        return {
            "is_nsfw": analysis["is_nsfw"],
            "nsfw_score": analysis["nsfw_score"],
            "detections": len(results),
            "body_coverage": analysis["body_coverage"],
            "clothing_analysis": analysis["clothing_analysis"],
            "detailed_description": analysis["description"],
            "detailed_analysis": analysis["detailed_analysis"],
            "detected_gender": analysis["detected_gender"],
            "confidence_score": analysis["confidence_score"],
            "detected_parts": analysis["detected_parts"],
            "coverage_percentage": analysis["coverage_percentage"]
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Analysis failed: {str(e)}")

@app.post("/analyze-video")
async def analyze_video(file: UploadFile = File(...)):
    if not file.content_type.startswith('video/'):
        raise HTTPException(status_code=400, detail="File must be a video")
    
    try:
        # Save uploaded file temporarily
        with tempfile.NamedTemporaryFile(delete=False, suffix='.mp4') as tmp_file:
            content = await file.read()
            tmp_file.write(content)
            tmp_file_path = tmp_file.name
        
        # Extract frames from video
        cap = cv2.VideoCapture(tmp_file_path)
        frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        # Sample 5 frames evenly distributed
        sample_frames = min(5, frame_count)
        frame_indices = [int(i * frame_count / sample_frames) for i in range(sample_frames)]
        
        frame_analyses = []
        max_nsfw_score = 0.0
        total_detections = 0
        
        for frame_idx in frame_indices:
            cap.set(cv2.CAP_PROP_POS_FRAMES, frame_idx)
            ret, frame = cap.read()
            if not ret:
                continue
                
            # Save frame temporarily
            with tempfile.NamedTemporaryFile(delete=False, suffix='.jpg') as frame_file:
                cv2.imwrite(frame_file.name, frame)
                
                # Analyze frame
                results = detector.detect(frame_file.name)
                total_detections += len(results)
                
                # Get detailed analysis for this frame
                frame_analysis = analyze_body_coverage(results)
                frame_analyses.append({
                    "frame": frame_idx,
                    "analysis": frame_analysis
                })
                
                max_nsfw_score = max(max_nsfw_score, frame_analysis["nsfw_score"])
                
                # Clean up frame file
                os.unlink(frame_file.name)
        
        cap.release()
        os.unlink(tmp_file_path)
        
        # Aggregate results from all frames
        is_nsfw = max_nsfw_score > 0.6 or any(frame["analysis"]["is_nsfw"] for frame in frame_analyses)
        
        # Find the most revealing frame for description
        most_revealing_frame = max(frame_analyses, key=lambda x: x["analysis"]["nsfw_score"]) if frame_analyses else None
        
        return {
            "is_nsfw": is_nsfw,
            "nsfw_score": round(max_nsfw_score, 3),
            "frames_analyzed": len(frame_indices),
            "total_detections": total_detections,
            "frame_analyses": frame_analyses,
            "summary": {
                "body_coverage": most_revealing_frame["analysis"]["body_coverage"] if most_revealing_frame else "Unknown",
                "clothing_analysis": most_revealing_frame["analysis"]["clothing_analysis"] if most_revealing_frame else "Unknown",
                "detailed_description": most_revealing_frame["analysis"]["description"] if most_revealing_frame else "No analysis available"
            }
        }
        
    except Exception as e:
        raise HTTPException(status_code=500, detail=f"Video analysis failed: {str(e)}")

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8001)