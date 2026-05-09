import os
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, send_file
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

def enhance_image_professional(image_bytes, scale=2, mode='default'):
    """Professional AI-like enhancement using advanced CV techniques"""
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image")
    
    # Get original dimensions
    h, w = img.shape[:2]
    
    # STEP 1: Noise Reduction
    denoised = cv2.fastNlMeansDenoisingColored(img, None, 7, 7, 7, 21)
    
    # STEP 2: Enhance details using unsharp masking
    gaussian = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    unsharp = cv2.addWeighted(denoised, 2.0, gaussian, -1.0, 0)
    
    # STEP 3: CLAHE for better contrast
    lab = cv2.cvtColor(unsharp, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(8, 8))
    l_enhanced = clahe.apply(l)
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # STEP 4: Face enhancement mode
    if mode == 'face':
        # Detect faces and enhance them
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.1, 5)
        
        for (fx, fy, fw, fh) in faces:
            face_roi = contrast_enhanced[fy:fy+fh, fx:fx+fw]
            # Enhance face specifically
            face_enhanced = cv2.detailEnhance(face_roi, sigma_s=15, sigma_r=0.25)
            face_enhanced = cv2.edgePreservingFilter(face_enhanced, flags=1, sigma_s=60, sigma_r=0.4)
            contrast_enhanced[fy:fy+fh, fx:fx+fw] = face_enhanced
    
    # STEP 5: Art/Anime mode
    elif mode == 'art':
        # Cartoonize effect
        bilateral = cv2.bilateralFilter(contrast_enhanced, 9, 75, 75)
        gray = cv2.cvtColor(contrast_enhanced, cv2.COLOR_BGR2GRAY)
        edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY, 9, 9)
        edges_colored = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
        contrast_enhanced = cv2.bitwise_and(bilateral, edges_colored)
    
    # STEP 6: Super Resolution-like upscaling
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Multi-step upscaling for better quality
    if scale > 2:
        # Step upscale
        mid_h = int(h * 2)
        mid_w = int(w * 2)
        upscaled = cv2.resize(contrast_enhanced, (mid_w, mid_h), interpolation=cv2.INTER_LANCZOS4)
        upscaled = cv2.resize(upscaled, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
    else:
        # Direct upscale with LANCZOS (highest quality)
        upscaled = cv2.resize(contrast_enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # STEP 7: Detail enhancement on upscaled image
    detail_enhanced = cv2.detailEnhance(upscaled, sigma_s=10, sigma_r=0.15)
    
    # STEP 8: Sharpening kernel
    sharpen_kernel = np.array([
        [-0.5, -1, -0.5],
        [-1, 7, -1],
        [-0.5, -1, -0.5]
    ]) / 3.0
    sharpened = cv2.filter2D(detail_enhanced, -1, sharpen_kernel)
    
    # STEP 9: Final color correction
    hsv = cv2.cvtColor(sharpened, cv2.COLOR_BGR2HSV)
    hsv[:, :, 1] = np.clip(hsv[:, :, 1] * 1.15, 0, 255).astype(np.uint8)  # Saturation boost
    hsv[:, :, 2] = np.clip(hsv[:, :, 2] * 1.05, 0, 255).astype(np.uint8)  # Brightness
    final = cv2.cvtColor(hsv, cv2.COLOR_HSV2BGR)
    
    # Encode to JPEG bytes
    _, img_encoded = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 95])
    
    return io.BytesIO(img_encoded.tobytes())

@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        if 'image' not in request.files:
            return {'error': 'No image'}, 400
        
        file = request.files['image']
        scale = int(request.form.get('scale', 2))
        mode = request.form.get('mode', 'default')
        
        # Limit to prevent abuse
        if scale > 4:
            scale = 4
        
        image_bytes = file.read()
        enhanced = enhance_image_professional(image_bytes, scale, mode)
        
        return send_file(enhanced, mimetype='image/jpeg')
        
    except Exception as e:
        return {'error': str(e)}, 500

@app.route('/')
def index():
    return {'status': 'AI Enhancement API Running'}

if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port)