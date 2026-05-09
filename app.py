import os
import io
import cv2
import numpy as np
from PIL import Image
from flask import Flask, request, send_file, jsonify
from flask_cors import CORS
import warnings
warnings.filterwarnings('ignore')

app = Flask(__name__)
CORS(app)

class AutoAIEnhancer:
    """Automatic AI Image Enhancer - Detects image type and applies best enhancement"""
    
    def __init__(self):
        self.face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        self.eye_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_eye.xml')
    
    def enhance(self, image_bytes):
        """Main enhancement pipeline - Fully Automatic"""
        
        # Load image
        nparr = np.frombuffer(image_bytes, np.uint8)
        img = cv2.imdecode(nparr, cv2.IMREAD_COLOR)
        
        if img is None:
            raise ValueError("Invalid image")
        
        original = img.copy()
        h, w = img.shape[:2]
        
        # ==========================================
        # AUTO-DETECT: What type of image is this?
        # ==========================================
        analysis = self.analyze_image(img)
        
        # Determine optimal scale based on image size
        optimal_scale = self.calculate_optimal_scale(h, w)
        
        # ==========================================
        # PHASE 1: AI PRE-PROCESSING
        # ==========================================
        img = self.ai_preprocess(img, analysis)
        
        # ==========================================
        # PHASE 2: NOISE REDUCTION
        # ==========================================
        img = self.ai_denoise(img, analysis['noise_level'])
        
        # ==========================================
        # PHASE 3: DETAIL RECOVERY
        # ==========================================
        img = self.ai_recover_details(img, analysis['blur_level'])
        
        # ==========================================
        # PHASE 4: COLOR & LIGHTING
        # ==========================================
        img = self.ai_color_correct(img, analysis)
        
        # ==========================================
        # PHASE 5: FACE DETECTION & ENHANCEMENT
        # ==========================================
        if analysis['has_faces']:
            img = self.ai_enhance_faces(img)
        
        # ==========================================
        # PHASE 6: SUPER RESOLUTION UPSCALE
        # ==========================================
        img = self.ai_upscale(img, optimal_scale)
        
        # ==========================================
        # PHASE 7: FINAL POLISH
        # ==========================================
        img = self.ai_final_polish(img)
        
        # Encode with maximum quality
        _, img_encoded = cv2.imencode('.jpg', img, [cv2.IMWRITE_JPEG_QUALITY, 98])
        
        return io.BytesIO(img_encoded.tobytes()), analysis, optimal_scale
    
    def analyze_image(self, img):
        """AI-powered image analysis"""
        h, w = img.shape[:2]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        
        # Detect noise level
        noise_level = self.detect_noise(gray)
        
        # Detect blur level
        blur_level = self.detect_blur(gray)
        
        # Detect brightness
        brightness = np.mean(gray)
        is_dark = brightness < 70
        is_bright = brightness > 180
        
        # Detect faces
        faces = self.face_cascade.detectMultiScale(gray, 1.1, 5, minSize=(40, 40))
        has_faces = len(faces) > 0
        
        # Detect if it's a document/text
        is_text = self.detect_text(img)
        
        # Detect image category
        category = self.categorize_image(img, gray, has_faces, is_text)
        
        return {
            'noise_level': noise_level,
            'blur_level': blur_level,
            'brightness': brightness,
            'is_dark': is_dark,
            'is_bright': is_bright,
            'has_faces': has_faces,
            'faces': faces,
            'is_text': is_text,
            'category': category,
            'height': h,
            'width': w
        }
    
    def detect_noise(self, gray):
        """Estimate noise level in image"""
        # Laplacian variance for noise estimation
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        noise_std = np.std(laplacian)
        # Normalize to 0-1 scale
        noise_level = min(noise_std / 50.0, 1.0)
        return noise_level
    
    def detect_blur(self, gray):
        """Detect blur level using variance of Laplacian"""
        laplacian = cv2.Laplacian(gray, cv2.CV_64F)
        variance = laplacian.var()
        # Higher variance = sharper image
        blur_level = max(0, min(1, 1 - (variance / 500)))
        return blur_level
    
    def detect_text(self, img):
        """Detect if image contains text"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        # Use MSER for text detection
        mser = cv2.MSER_create()
        regions, _ = mser.detectRegions(gray)
        # If many small regions, likely text
        return len(regions) > 30
    
    def categorize_image(self, img, gray, has_faces, is_text):
        """Categorize image type"""
        if is_text:
            return 'document'
        if has_faces:
            return 'portrait'
        
        # Check color variance
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV)
        saturation = np.mean(hsv[:, :, 1])
        
        if saturation < 30:
            return 'low_color'
        if saturation > 120:
            return 'vibrant'
        
        # Edge density for detail level
        edges = cv2.Canny(gray, 50, 150)
        edge_density = np.mean(edges) / 255
        
        if edge_density > 0.3:
            return 'detailed'
        elif edge_density > 0.1:
            return 'normal'
        else:
            return 'smooth'
    
    def calculate_optimal_scale(self, h, w):
        """Calculate best upscale ratio"""
        pixels = h * w
        
        if pixels < 50000:  # Very small (< ~224x224)
            return 4
        elif pixels < 200000:  # Small (< ~450x450)
            return 3
        elif pixels < 500000:  # Medium (< ~700x700)
            return 2
        elif pixels < 1000000:  # Large (< ~1000x1000)
            return 2
        else:  # Very large
            return 1  # Don't upscale, just enhance
    
    def ai_preprocess(self, img, analysis):
        """AI-based pre-processing"""
        if analysis['is_dark']:
            # Gamma correction for dark images
            gamma = 1.5
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
        
        if analysis['is_bright']:
            # Reduce overexposure
            gamma = 0.8
            inv_gamma = 1.0 / gamma
            table = np.array([((i / 255.0) ** inv_gamma) * 255 for i in range(256)]).astype(np.uint8)
            img = cv2.LUT(img, table)
        
        return img
    
    def ai_denoise(self, img, noise_level):
        """Adaptive noise reduction"""
        if noise_level > 0.6:
            # Heavy noise - strong denoising
            img = cv2.fastNlMeansDenoisingColored(img, None, 10, 10, 7, 21)
            img = cv2.bilateralFilter(img, 9, 75, 75)
        elif noise_level > 0.3:
            # Medium noise
            img = cv2.fastNlMeansDenoisingColored(img, None, 5, 5, 7, 21)
            img = cv2.bilateralFilter(img, 7, 50, 50)
        elif noise_level > 0.1:
            # Light noise
            img = cv2.bilateralFilter(img, 5, 50, 50)
        
        return img
    
    def ai_recover_details(self, img, blur_level):
        """AI detail recovery based on blur level"""
        if blur_level > 0.5:
            # Very blurry - strong detail recovery
            # Multi-scale detail enhancement
            g1 = cv2.GaussianBlur(img, (0, 0), 1.0)
            g2 = cv2.GaussianBlur(img, (0, 0), 3.0)
            g3 = cv2.GaussianBlur(img, (0, 0), 6.0)
            
            d1 = cv2.subtract(img, g1)
            d2 = cv2.subtract(g1, g2)
            d3 = cv2.subtract(g2, g3)
            
            img = cv2.addWeighted(img, 1.0, d1, 0.8, 0)
            img = cv2.addWeighted(img, 1.0, d2, 0.5, 0)
            img = cv2.addWeighted(img, 1.0, d3, 0.3, 0)
            
            # Sharpening
            kernel = np.array([
                [-0.5, -1.0, -0.5],
                [-1.0, 7.0, -1.0],
                [-0.5, -1.0, -0.5]
            ]) / 2.0
            img = cv2.filter2D(img, -1, kernel)
            
        elif blur_level > 0.2:
            # Slightly blurry - mild detail enhancement
            gaussian = cv2.GaussianBlur(img, (0, 0), 2.0)
            detail = cv2.subtract(img, gaussian)
            img = cv2.addWeighted(img, 1.0, detail, 0.4, 0)
            
            # Light sharpen
            kernel = np.array([
                [0, -0.3, 0],
                [-0.3, 2.2, -0.3],
                [0, -0.3, 0]
            ])
            img = cv2.filter2D(img, -1, kernel)
        
        return img
    
    def ai_color_correct(self, img, analysis):
        """Intelligent color correction"""
        # Auto white balance
        img = self.auto_white_balance(img)
        
        # CLAHE contrast enhancement
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        l, a, b = cv2.split(lab)
        
        clip_limit = 2.0
        if analysis['is_dark']:
            clip_limit = 3.0
        elif analysis['is_bright']:
            clip_limit = 1.5
        
        clahe = cv2.createCLAHE(clipLimit=clip_limit, tileGridSize=(8, 8))
        l = clahe.apply(l)
        
        img = cv2.merge((l, a, b))
        img = cv2.cvtColor(img, cv2.COLOR_LAB2BGR)
        
        # Saturation & vibrance
        hsv = cv2.cvtColor(img, cv2.COLOR_BGR2HSV).astype(np.float32)
        
        # Smart saturation boost
        s = hsv[:, :, 1]
        s_mean = np.mean(s)
        
        if s_mean < 50:
            boost = 25
        elif s_mean < 80:
            boost = 15
        else:
            boost = 5
        
        hsv[:, :, 1] = np.clip(s + boost, 0, 255)
        
        # Brightness adjustment
        v = hsv[:, :, 2]
        v_mean = np.mean(v)
        
        if v_mean < 80:
            v_boost = 20
        elif v_mean > 160:
            v_boost = -10
        else:
            v_boost = 0
        
        hsv[:, :, 2] = np.clip(v + v_boost, 0, 255)
        
        img = cv2.cvtColor(hsv.astype(np.uint8), cv2.COLOR_HSV2BGR)
        
        return img
    
    def auto_white_balance(self, img):
        """Automatic white balance correction"""
        result = img.copy().astype(np.float32)
        
        # Calculate channel means
        b_mean = np.mean(result[:, :, 0])
        g_mean = np.mean(result[:, :, 1])
        r_mean = np.mean(result[:, :, 2])
        
        # Calculate global mean
        global_mean = (b_mean + g_mean + r_mean) / 3.0
        
        # Apply gains
        if g_mean > 0:
            result[:, :, 0] = np.clip(result[:, :, 0] * (global_mean / max(b_mean, 1)), 0, 255)
            result[:, :, 2] = np.clip(result[:, :, 2] * (global_mean / max(r_mean, 1)), 0, 255)
        
        return result.astype(np.uint8)
    
    def ai_enhance_faces(self, img):
        """Professional face enhancement"""
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
        faces = self.face_cascade.detectMultiScale(gray, 1.05, 5, minSize=(30, 30))
        
        for (fx, fy, fw, fh) in faces:
            # Expand slightly
            fx = max(0, fx - int(fw * 0.15))
            fy = max(0, fy - int(fh * 0.2))
            fw = min(img.shape[1] - fx, int(fw * 1.3))
            fh = min(img.shape[0] - fy, int(fh * 1.4))
            
            face_roi = img[fy:fy+fh, fx:fx+fw]
            
            # Skin smoothing
            face_roi = cv2.edgePreservingFilter(face_roi, flags=2, sigma_s=60, sigma_r=0.2)
            face_roi = cv2.detailEnhance(face_roi, sigma_s=20, sigma_r=0.15)
            
            # Eye enhancement
            face_gray = gray[fy:fy+fh, fx:fx+fw]
            eyes = self.eye_cascade.detectMultiScale(face_gray, 1.05, 3, minSize=(10, 10))
            
            for (ex, ey, ew, eh) in eyes:
                eye_roi = face_roi[ey:ey+eh, ex:ex+ew]
                # Sharpen eyes
                kernel = np.array([[0, -0.5, 0], [-0.5, 3.0, -0.5], [0, -0.5, 0]])
                face_roi[ey:ey+eh, ex:ex+ew] = cv2.filter2D(eye_roi, -1, kernel)
            
            # Blend face back smoothly
            mask = np.zeros((fh, fw), dtype=np.float32)
            cv2.ellipse(mask, (fw//2, fh//2), (fw//2, fh//2), 0, 0, 360, 1, -1)
            mask = cv2.GaussianBlur(mask, (fw//3 * 2 + 1, fh//3 * 2 + 1), 0)
            mask = np.stack([mask] * 3, axis=2)
            
            blended = (face_roi * mask + img[fy:fy+fh, fx:fx+fw] * (1 - mask))
            img[fy:fy+fh, fx:fx+fw] = blended.astype(np.uint8)
        
        return img
    
    def ai_upscale(self, img, scale):
        """High quality AI upscaling"""
        if scale <= 1:
            return img
        
        h, w = img.shape[:2]
        
        # For larger scales, use step upscaling
        if scale >= 3:
            # First step: 2x
            mid_h, mid_w = h * 2, w * 2
            img = cv2.resize(img, (mid_w, mid_h), interpolation=cv2.INTER_LANCZOS4)
            img = cv2.detailEnhance(img, sigma_s=3, sigma_r=0.1)
            
            remaining = scale / 2
            if remaining > 1:
                new_h, new_w = int(mid_h * remaining), int(mid_w * remaining)
                img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_CUBIC)
        else:
            new_h, new_w = int(h * scale), int(w * scale)
            img = cv2.resize(img, (new_w, new_h), interpolation=cv2.INTER_LANCZOS4)
        
        # Detail enhancement after upscaling
        img = cv2.detailEnhance(img, sigma_s=8, sigma_r=0.12)
        
        return img
    
    def ai_final_polish(self, img):
        """Final AI polish for perfect output"""
        # Light edge-aware smoothing
        img = cv2.edgePreservingFilter(img, flags=1, sigma_s=30, sigma_r=0.1)
        
        # Final micro-sharpening
        kernel = np.array([
            [0, -0.15, 0],
            [-0.15, 1.6, -0.15],
            [0, -0.15, 0]
        ])
        img = cv2.filter2D(img, -1, kernel)
        
        # Ensure valid range
        img = np.clip(img, 0, 255).astype(np.uint8)
        
        return img


# Initialize enhancer
enhancer = AutoAIEnhancer()


@app.route('/enhance', methods=['POST'])
def enhance():
    """Auto AI Enhancement Endpoint"""
    try:
        if 'image' not in request.files:
            return jsonify({'error': 'No image uploaded', 'status': 'failed'}), 400
        
        file = request.files['image']
        
        if file.filename == '':
            return jsonify({'error': 'Empty filename', 'status': 'failed'}), 400
        
        # Read image
        image_bytes = file.read()
        
        # Check size (max 25MB)
        if len(image_bytes) > 25 * 1024 * 1024:
            return jsonify({'error': 'File too large. Max 25MB', 'status': 'failed'}), 400
        
        # Auto enhance
        enhanced, analysis, scale = enhancer.enhance(image_bytes)
        
        return send_file(
            enhanced,
            mimetype='image/jpeg',
            as_attachment=True,
            download_name=f'AI_Enhanced_{analysis["category"]}_{scale}x.jpg'
        )
        
    except ValueError as ve:
        return jsonify({'error': str(ve), 'status': 'failed'}), 400
    except Exception as e:
        return jsonify({'error': f'Enhancement failed: {str(e)}', 'status': 'failed'}), 500


@app.route('/')
def index():
    return jsonify({
        'status': 'running',
        'service': 'Auto AI Image Enhancer v3.0',
        'features': [
            'Auto-detect image type',
            'Auto noise reduction',
            'Auto detail recovery',
            'Auto color correction',
            'Auto face detection & enhancement',
            'Auto optimal upscaling',
            'Full HD quality output'
        ],
        'endpoints': {
            '/enhance': 'POST - Upload image for automatic AI enhancement',
            '/health': 'GET - Health check'
        }
    })


@app.route('/health')
def health():
    return jsonify({'status': 'healthy'})


if __name__ == '__main__':
    port = int(os.environ.get('PORT', 5000))
    app.run(host='0.0.0.0', port=port, debug=False)
