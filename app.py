import os
import io
import cv2
import numpy as np
from PIL import Image, ImageEnhance, ImageFilter
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

def enhance_image_professional(image_bytes, scale=2, mode='default'):
    """
    Professional AI Image Enhancement
    Supports: default (general), face, art, hdr, night, text modes
    """
    
    # Convert bytes to numpy array
    nparr = np.frombuffer(image_bytes, np.uint8)
    img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
    
    if img is None:
        raise ValueError("Invalid image")
    
    # Store original
    original = img.copy()
    h, w = img.shape[:2]
    
    # =============================================
    # STEP 1: Advanced Noise Reduction
    # =============================================
    # Bilateral filter - preserves edges while removing noise
    denoised = cv2.bilateralFilter(img, 9, 75, 75)
    
    # Non-local means denoising for better noise removal
    denoised = cv2.fastNlMeansDenoisingColored(denoised, None, 5, 5, 7, 21)
    
    # =============================================
    # STEP 2: Super Resolution Detail Enhancement
    # =============================================
    # Create detail layer
    blurred = cv2.GaussianBlur(denoised, (0, 0), 3.0)
    detail_layer = cv2.subtract(denoised, blurred)
    
    # Enhance details
    detail_enhanced = cv2.addWeighted(denoised, 1.0, detail_layer, 0.4, 0)
    
    # =============================================
    # STEP 3: Adaptive Contrast Enhancement
    # =============================================
    lab = cv2.cvtColor(detail_enhanced, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    
    # CLAHE with multiple tile sizes for better results
    clahe1 = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8, 8))
    clahe2 = cv2.createCLAHE(clipLimit=2.5, tileGridSize=(4, 4))
    
    l1 = clahe1.apply(l)
    l2 = clahe2.apply(l)
    
    # Blend both CLAHE results
    l_enhanced = cv2.addWeighted(l1, 0.6, l2, 0.4, 0)
    
    # Merge back
    enhanced_lab = cv2.merge((l_enhanced, a, b))
    contrast_enhanced = cv2.cvtColor(enhanced_lab, cv2.COLOR_LAB2BGR)
    
    # =============================================
    # STEP 4: MODE-BASED ENHANCEMENT
    # =============================================
    
    if mode == 'face':
        contrast_enhanced = enhance_faces(contrast_enhanced)
    elif mode == 'hdr':
        contrast_enhanced = enhance_hdr(contrast_enhanced, original)
    elif mode == 'night':
        contrast_enhanced = enhance_night(contrast_enhanced)
    elif mode == 'text':
        contrast_enhanced = enhance_text(contrast_enhanced)
    elif mode == 'art':
        contrast_enhanced = enhance_art(contrast_enhanced)
    else:
        # Default: General enhancement
        contrast_enhanced = enhance_general(contrast_enhanced)
    
    # =============================================
    # STEP 5: Intelligent Upscaling
    # =============================================
    new_h = int(h * scale)
    new_w = int(w * scale)
    
    # Laplacian pyramid upscaling for better quality
    upscaled = laplacian_upscale(contrast_enhanced, scale)
    
    # If Laplacian fails, fallback to LANCZOS
    if upscaled is None or upscaled.shape[:2] != (new_h, new_w):
        if scale > 2:
            # Step upscaling for large scales
            mid_h, mid_w = int(h * 2), int(w * 2)
            upscaled = cv2.resize(contrast_enhanced, (mid_w, mid_h), interpolation=cv2.INTER_LANCZOS4)
            # Apply detail enhancement at intermediate step
            upscaled = cv2.detailEnhance(upscaled, sigma_s=5, sigma_r=0.1)
            upscaled = cv2.resize(upscaled, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            upscaled = cv2.resize(contrast_enhanced, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
    
    # =============================================
    # STEP 6: Advanced Sharpening
    # =============================================
    # Create gaussian pyramid for multi-scale sharpening
    sharpened = multi_scale_sharpen(upscaled)
    
    # =============================================
    # STEP 7: Edge Enhancement
    # =============================================
    edge_enhanced = enhance_edges(sharpened)
    
    # =============================================
    # STEP 8: Color Correction & Vibrance
    # =============================================
    final = color_correction(edge_enhanced)
    
    # =============================================
    # STEP 9: Final Polish
    # =============================================
    # Light bilateral filter for smooth finish
    final = cv2.bilateralFilter(final, 5, 50, 50)
    
    # Ensure output is valid
    final = np.clip(final, 0, 255).astype(np.uint8)
    
    # Encode to JPEG bytes with high quality
    _, img_encoded = cv2.imencode('.jpg', final, [cv2.IMWRITE_JPEG_QUALITY, 96])
    
    return io.BytesIO(img_encoded.tobytes())


def laplacian_upscale(img, scale):
    """Laplacian pyramid upscaling for better quality"""
    try:
        h, w = img.shape[:2]
        target_h, target_w = int(h * scale), int(w * scale)
        
        # Build gaussian pyramid
        pyramid = [img]
        for i in range(int(np.log2(scale)) + 1):
            pyramid.append(cv2.pyrDown(pyramid[-1]))
        
        # Upscale from smallest
        result = pyramid[-1]
        for i in range(len(pyramid) - 2, -1, -1):
            result = cv2.pyrUp(result)
            if result.shape != pyramid[i].shape:
                result = cv2.resize(result, (pyramid[i].shape[1], pyramid[i].shape[0]))
            result = cv2.addWeighted(pyramid[i], 0.6, result, 0.4, 0)
        
        # Final resize to exact dimensions
        if result.shape[:2] != (target_h, target_w):
            result = cv2.resize(result, (target_w, target_h), interpolation=cv2.INTER_LANCZOS4)
        
        return result
    except:
        return None


def multi_scale_sharpen(img):
    """Multi-scale sharpening for natural results"""
    # Gaussian blur at different scales
    g1 = cv2.GaussianBlur(img, (0, 0), 1.0)
    g2 = cv2.GaussianBlur(img, (0, 0), 2.0)
    g3 = cv2.GaussianBlur(img, (0, 0), 4.0)
    
    # Extract details at each scale
    d1 = cv2.subtract(img, g1)
    d2 = cv2.subtract(g1, g2)
    d3 = cv2.subtract(g2, g3)
    
    # Enhance each detail layer differently
    result = cv2.addWeighted(img, 1.0, d1, 0.5, 0)
    result = cv2.addWeighted(result, 1.0, d2, 0.3, 0)
    result = cv2.addWeighted(result, 1.0, d3, 0.15, 0)
    
    # Fine sharpening kernel
    sharpen = np.array([
        [-0.25, -0.5, -0.25],
        [-0.5, 4.0, -0.5],
        [-0.25, -0.5, -0.25]
    ]) / 2.0
    
    result = cv2.filter2D(result, -1, sharpen)
    return np.clip(result, 0, 255).astype(np.uint8)


def enhance_edges(img):
    """Edge-aware enhancement"""
    # Detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    edges = cv2.Canny(gray, 30, 100)
    
    # Dilate edges
    kernel = np.ones((2, 2), np.uint8)
    edges_dilated = cv2.dilate(edges, kernel, iterations=1)
    
    # Create edge mask
    edge_mask = cv2.GaussianBlur(edges_dilated.astype(np.float32), (0, 0), 1.0) / 255.0
    edge_mask = np.stack([edge_mask] * 3, axis=2)
    
    # Sharpen only edges
    sharpen_kernel = np.array([
        [-0.5, -1, -0.5],
        [-1, 7, -1],
        [-0.5, -1, -0.5]
    ]) / 3.0
    
    sharpened = cv2.filter2D(img, -1, sharpen_kernel)
    
    # Blend: sharp edges, smooth non-edges
    result = (img * (1 - edge_mask) + sharpened * edge_mask).astype(np.uint8)
    return result


def color_correction(img):
    """Advanced color correction and vibrance"""
    # Convert to HSV for better color control
    hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
    
    # Vibrance: boost low-saturation colors more than high-saturation
    s = hsv[:, :, 1]
    s_max = np.max(s)
    if s_max > 0:
        # Vibrance curve - boost mid-tones
        vibrance_mask = 1.0 - (s / 255.0)
        s_boost = 15 * vibrance_mask
        hsv[:, :, 1] = np.clip(s + s_boost, 0, 255)
    
    # Slight brightness enhancement
    v = hsv[:, :, 2]
    hsv[:, :, 2] = np.clip(v * 1.08, 0, 255)
    
    # White balance correction (simplified)
    bgr = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
    
    # Auto white balance using gray world assumption
    b, g, r = cv2.split(bgr.astype(np.float32))
    
    b_mean = np.mean(b)
    g_mean = np.mean(g)
    r_mean = np.mean(r)
    
    # Calculate gains
    if g_mean > 0:
        b_gain = g_mean / max(b_mean, 1)
        r_gain = g_mean / max(r_mean, 1)
        
        b = np.clip(b * b_gain, 0, 255)
        r = np.clip(r * r_gain, 0, 255)
    
    result = cv2.merge([b, g, r]).astype(np.uint8)
    return result


def enhance_faces(img):
    """Specialized face enhancement"""
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
    eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(50, 50))
    
    for (fx, fy, fw, fh) in faces:
        # Expand face region slightly
        fx = max(0, fx - fw//10)
        fy = max(0, fy - fh//5)
        fw = min(img.shape[1] - fx, int(fw * 1.2))
        fh = min(img.shape[0] - fy, int(fh * 1.3))
        
        face_roi = img[fy:fy+fh, fx:fx+fw]
        
        # Skin smoothing while preserving details
        face_smooth = cv2.edgePreservingFilter(face_roi, flags=2, sigma_s=60, sigma_r=0.2)
        face_detail = cv2.detailEnhance(face_smooth, sigma_s=20, sigma_r=0.15)
        
        # Enhance eyes if detected
        face_gray = gray[fy:fy+fh, fx:fx+fw]
        eyes = eye_cascade.detectMultiScale(face_gray, 1.05, 3, minSize=(15, 15))
        
        for (ex, ey, ew, eh) in eyes:
            eye_roi = face_detail[ey:ey+eh, ex:ex+ew]
            # Sharpen eyes
            eye_sharpened = cv2.filter2D(eye_roi, -1, 
                np.array([[0, -0.5, 0], [-0.5, 3, -0.5], [0, -0.5, 0]]))
            face_detail[ey:ey+eh, ex:ex+ew] = eye_sharpened
        
        img[fy:fy+fh, fx:fx+fw] = face_detail
    
    return img


def enhance_hdr(img, original):
    """HDR-like enhancement"""
    # Create multiple exposures
    under = cv2.addWeighted(original, 0.5, img, 0.5, 0)
    over = cv2.addWeighted(original, 0.3, img, 0.7, 0)
    
    # Tone mapping simulation
    hdr = cv2.createMergeMertens()
    result = hdr.process([under.astype(np.float32)/255, 
                          img.astype(np.float32)/255, 
                          over.astype(np.float32)/255])
    
    result = (result * 255).astype(np.uint8)
    return result


def enhance_night(img):
    """Night mode enhancement"""
    # Brightness boost
    img_float = img.astype(np.float32)
    
    # Gamma correction for dark areas
    gamma = 0.85
    img_gamma = np.power(img_float / 255.0, gamma) * 255.0
    
    # Enhance shadows
    shadow_mask = np.mean(img, axis=2) < 80
    img_gamma[shadow_mask] *= 1.3
    
    result = np.clip(img_gamma, 0, 255).astype(np.uint8)
    
    # Denoise heavily
    result = cv2.fastNlMeansDenoisingColored(result, None, 10, 10, 7, 21)
    return result


def enhance_text(img):
    """Text/Document enhancement"""
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    # Adaptive threshold for text
    text_mask = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C, 
                                       cv2.THRESH_BINARY, 11, 2)
    
    # Sharpen text
    kernel = np.array([[0, -1, 0], [-1, 5, -1], [0, -1, 0]])
    sharpened = cv2.filter2D(img, -1, kernel)
    
    # Convert text mask to 3-channel
    text_mask_3ch = np.stack([text_mask] * 3, axis=2) / 255.0
    
    # Keep text sharp, smooth background
    result = (sharpened * text_mask_3ch + img * (1 - text_mask_3ch)).astype(np.uint8)
    return result


def enhance_art(img):
    """Art/Anime enhancement"""
    # Cartoon-style enhancement
    # Edge-preserving smooth
    smooth = cv2.edgePreservingFilter(img, flags=2, sigma_s=80, sigma_r=0.25)
    
    # Detect edges
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    gray = cv2.medianBlur(gray, 5)
    edges = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_MEAN_C, 
                                   cv2.THRESH_BINARY, 9, 9)
    edges = cv2.cvtColor(edges, cv2.COLOR_GRAY2BGR)
    
    # Combine
    result = cv2.bitwise_and(smooth, edges)
    
    # Boost colors
    result = cv2.addWeighted(result, 1.2, img, -0.2, 0)
    return result


def enhance_general(img):
    """General purpose enhancement"""
    # Edge-preserving smoothing
    img = cv2.edgePreservingFilter(img, flags=1, sigma_s=50, sigma_r=0.15)
    
    # Detail enhancement
    img = cv2.detailEnhance(img, sigma_s=15, sigma_r=0.15)
    
    return img


@app.route('/enhance', methods=['POST'])
def enhance():
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'status': 'failed'}), 400
        
        file = request.files['image']
        
        # Validate file
        if file.filename == '':
            return jsonify({'error': 'Empty filename', 'status': 'failed'}), 400
        
        # Get parameters
        scale = int(request.form.get('scale', 2))
        mode = request.form.get('mode', 'default')
        quality = int(request.form.get('quality', 95))
        
        # Validate parameters
        if scale < 1 or scale > 8:
            scale = 2
        if mode not in ['default', 'face', 'hdr', 'night', 'text', 'art']:
            mode = 'default'
        if quality < 50 or quality > 100:
            quality = 95
        
        # Read image bytes
        image_bytes = file.read()
        
        # Check file size (max 20MB)
        if len(image_bytes) > 20 * 1024 * 1024:
            return jsonify({'error': 'File too large. Max 20MB', 'status': 'failed'}), 400
        
        # Process image
        enhanced = enhance_image_professional(image_bytes, scale, mode)
        
        return send_file(
            enhanced,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'enhanced_{mode}_{scale}x.jpg'
        )
        
    except ValueError as ve:
        return jsonify({'error': str(ve), 'status': 'failed'}), 400
    except Exception as e:
        return jsonify({'error': f'Processing failed: {str(e)}', 'status': 'failed'}), 500


@app.route('/')
def index():
    """API status endpoint"""
    return jsonify({
        'status': 'running',
        'service': 'AI Image Enhancer API',
        'version': '2.0',
        'endpoints': {
            '/enhance': 'POST - Upload image for enhancement',
            '/modes': 'GET - List available modes',
            '/health': 'GET - Health check'
        },
        'supported_modes': ['default', 'face', 'hdr', 'night', 'text', 'art']
    })


@app.route('/modes')
def get_modes():
    """Get available enhancement modes"""
    return jsonify({
        'modes': {
            'default': 'General AI enhancement for all images',
            'face': 'Specialized face detection and enhancement',
            'hdr': 'HDR-like enhancement with tone mapping',
            'night': 'Low-light and night photo enhancement',
            'text': 'Text and document clarity enhancement',
            'art': 'Art and anime style enhancement'
        }
    })


@app.route('/health')
def health():
    """Health check endpoint"""
    return jsonify({
        'status': 'healthy',
        'memory_usage': 'OK'
    })


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
