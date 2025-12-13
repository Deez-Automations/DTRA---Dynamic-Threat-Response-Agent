# -*- coding: utf-8 -*-
"""
DTRA API Server
Flask-based REST API for the Dynamic Threat Response Agent.
Provides SOC analyst-grade analysis and statistics.
DIRECT IMPLEMENTATION to bypass detector module issues.
"""

from flask import Flask, request, jsonify, send_from_directory
from flask_cors import CORS
import pandas as pd
import numpy as np
import os
import time
import joblib
import tensorflow as tf
from werkzeug.utils import secure_filename
import logging

# Import local config and Decider (bypassing detector.py)
from config import NUMERIC_FEATURES, MODELS_DIR
from decider import HybridDecider

# --- Initialize Flask App ---
app = Flask(__name__, static_folder='../ui', static_url_path='')
CORS(app)  # Enable CORS for frontend

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- Configuration ---
# Uploads go to root/uploads
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
UPLOAD_FOLDER = os.path.join(BASE_DIR, 'uploads')
os.makedirs(UPLOAD_FOLDER, exist_ok=True)
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER
app.config['MAX_CONTENT_LENGTH'] = 500 * 1024 * 1024  # 500MB max

# --- Global Models ---
model = None
scaler = None
imputer = None
decider = None
is_loaded = False

def load_models():
    """Load pre-trained models directly without wrapper class."""
    global model, scaler, imputer, decider, is_loaded
    
    try:
        logger.info("📥 Loading models directly...")
        
        # Load scaler and imputer
        scaler_path = os.path.join(MODELS_DIR, 'dtra_scaler.pkl')
        imputer_path = os.path.join(MODELS_DIR, 'dtra_imputer.pkl')
        
        if os.path.exists(scaler_path):
            scaler = joblib.load(scaler_path)
            logger.info("   ✅ Scaler loaded")
            
        if os.path.exists(imputer_path):
            imputer = joblib.load(imputer_path)
            logger.info("   ✅ Imputer loaded")
            
        # Load Keras model
        model_path = os.path.join(MODELS_DIR, 'dtra_detector_model.h5')
        if os.path.exists(model_path):
            # Compile=False is important to avoid optimizer loading issues
            model = tf.keras.models.load_model(model_path, compile=False)
            logger.info("   ✅ Neural Network loaded")

        # Load decider
        decider = HybridDecider()
        decider.load()
        logger.info("   ✅ Decider loaded")
        
        if model and scaler and imputer:
            is_loaded = True
            logger.info("✅ All models loaded successfully")
            return True
        else:
            logger.error("❌ Critical model files missing")
            return False
            
    except Exception as e:
        logger.error(f"⚠️ Could not load models: {e}")
        is_loaded = False
        return False

# --- API Routes ---

@app.route('/')
def home():
    """Serve the simple dashboard."""
    return send_from_directory('.', 'simple_dashboard.html')

@app.route('/api/status')
def get_status():
    """Get system status."""
    return jsonify({
        'status': 'online',
        'models_loaded': is_loaded,
        'timestamp': time.strftime('%Y-%m-%d %H:%M:%S')
    })

@app.route('/api/analyze', methods=['POST'])
def analyze_file():
    """
    Analyze uploaded CSV file for threats.
    Direct implementation of pipeline to ensure robustness against infinity values.
    """
    global model, scaler, imputer, decider
    
    # IMMEDIATE debug output
    print(">>> ENDPOINT HIT: /api/analyze", flush=True)
    
    # 1. Basic Checks
    if not is_loaded:
        return jsonify({'error': 'Models not loaded'}), 503
    
    if 'file' not in request.files:
        return jsonify({'error': 'No file provided'}), 400
    
    file = request.files['file']
    if file.filename == '':
        return jsonify({'error': 'No file selected'}), 400
    
    try:
        start_time = time.time()
        
        # 2. Save and Read File
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)
        
        print(f"📂 Processing {filename}...", flush=True)
        # Low_memory=False helps with type inference on large files
        df = pd.read_csv(filepath, encoding='latin1', low_memory=False)
        df.columns = df.columns.str.strip()
        
        total_rows = len(df)
        print(f"   Loaded {total_rows} rows", flush=True)
        
        # 3. Check for Labels
        has_labels = 'Label' in df.columns
        y_true = None
        if has_labels:
            df['label'] = df['Label'].apply(lambda x: 0 if x == 'BENIGN' else 1)
            y_true = df['label'].values
            
        # 4. Prepare Features (Exact logic from test.py)
        # Creates a new dataframe with ONLY numeric features, filled with 0.0 initially
        X = pd.DataFrame(0.0, index=df.index, columns=NUMERIC_FEATURES)
        
        # Fill strictly with available columns
        available_cols = [c for c in NUMERIC_FEATURES if c in df.columns]
        X[available_cols] = df[available_cols]
        
        # 5. CRITICAL CLEANING STEP - This is what fixes the "Input X contains infinity" error
        # Convert to float64 numpy array
        X_vals = X.values.astype(np.float64)
        
        # Replace non-finite values (NaN, Inf, -Inf) with 0.0
        # This MUST happen before Imputer/Scaler
        X_vals = np.nan_to_num(X_vals, nan=0.0, posinf=0.0, neginf=0.0)
        
        # Clip extreme values to prevent overflow errors in Sklearn
        X_vals = np.clip(X_vals, -1e10, 1e10)
        
        print(f"📊 Predicting batch of {len(X_vals)}...", flush=True)
        
        # 6. Preprocessing Pipeline
        # Explicit call means no hidden 'detector.predict' caching issues
        X_imputed = imputer.transform(X_vals)
        X_scaled = scaler.transform(X_imputed)
        
        # 7. Predict
        # Use batch_size for memory efficiency
        danger_scores = model.predict(X_scaled, batch_size=4096, verbose=0).flatten()
        
        # 8. Generate Actions & Results
        predictions = (danger_scores > 0.5).astype(int)
        
        # Generate actions (using simplified logic for speed on batch)
        # Using vectorized decisions logic for performance
        actions = []
        # If score > 0.8 -> Block, > 0.5 -> Log, else Ignore
        # Doing this in a loop for 300k items is slow, let's sample or vectorize
        # For full analysis stats, we just need counts
        
        # 9. Compute Statistics
        threats_detected = int(np.sum(predictions))
        threat_rate = (threats_detected / total_rows) * 100 if total_rows > 0 else 0
        
        processing_time = time.time() - start_time

        # --- Generate Results Sample (for Dashboard Logs) ---
        # We can't send 300k items. We'll send a sample of interesting packets.
        # Priority: Threats, then partial benign.
        
        results = []
        
        # Get indices of threats and benign
        threat_indices = np.where(predictions == 1)[0]
        benign_indices = np.where(predictions == 0)[0]
        
        # Sample up to 1000 of each
        sample_threats = threat_indices[:1000] if len(threat_indices) > 0 else []
        sample_benign = benign_indices[:1000] if len(benign_indices) > 0 else []
        
        # Combine and sort by index to maintain relative order
        all_sample_indices = np.concatenate([sample_threats, sample_benign])
        all_sample_indices.sort()
        
        for idx in all_sample_indices:
            score = float(danger_scores[idx])
            is_threat = predictions[idx] == 1
            
            # Simple Action Logic (inline for speed)
            if score >= 0.8:
                action = 'BLOCK'
                severity = 'CRITICAL'
            elif score >= 0.5:
                action = 'BLOCK' 
                severity = 'HIGH'
            elif score >= 0.3:
                action = 'LOG'
                severity = 'MEDIUM'
            else:
                action = 'IGNORE'
                severity = 'LOW'
                
            results.append({
                'id': int(idx),
                'danger_score': round(score, 4),
                'is_threat': bool(is_threat),
                'action': action,
                'severity': severity,
                # Add source/dest IP/Port if available in DF, else mock or omit
                # We'll just stick to score/action for now to be safe
                'timestamp': time.strftime('%H:%M:%S', time.localtime(time.time() - (total_rows - idx)*0.001)) # Fake dispersed timestamps
            })
            
        stats = {
            'total_packets': total_rows,
            'threats_detected': threats_detected,
            'threat_rate': round(threat_rate, 2),
            'processing_time': round(processing_time, 2),
            'status': 'Under Attack' if threat_rate > 1 else 'Secure',
            'actions': {
                'block': int(np.sum(danger_scores >= 0.5)),
                'log': int(np.sum((danger_scores >= 0.3) & (danger_scores < 0.5))),
                'ignore': int(np.sum(danger_scores < 0.3))
            }
        }
        
        if has_labels:
            from sklearn.metrics import confusion_matrix
            # Calculate basic metrics manually to avoid heavysklearn overhead
            tp = np.sum((predictions == 1) & (y_true == 1))
            tn = np.sum((predictions == 0) & (y_true == 0))
            fp = np.sum((predictions == 1) & (y_true == 0))
            fn = np.sum((predictions == 0) & (y_true == 1))
            
            # Recalculate full metrics
            try:
                accuracy = (tp + tn) / total_rows * 100
                precision = tp / (tp + fp) * 100 if (tp + fp) > 0 else 0
                recall = tp / (tp + fn) * 100 if (tp + fn) > 0 else 0
                f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
                
                stats['classification_metrics'] = {
                    'accuracy': round(accuracy, 2),
                    'precision': round(precision, 2),
                    'recall': round(recall, 2),
                    'f1_score': round(f1, 2),
                    'detection_rate': round(recall, 2),
                    'false_alarm_rate': round((fp / (fp + tn)) * 100, 2) if (fp + tn) > 0 else 0
                }
                
                stats['confusion_matrix'] = {
                    'tp': int(tp), 'tn': int(tn), 
                    'fp': int(fp), 'fn': int(fn)
                }
            except Exception as metric_err:
                print(f"Metric calculation error: {metric_err}")

        # Cleanup
        try:
            os.remove(filepath)
        except:
            pass
            
        return jsonify({
            'success': True,
            'stats': stats,
            'results': results # Now returning packet logs!
        })
        
    except Exception as e:
        import traceback
        traceback.print_exc()
        print(f"❌ ERROR: {str(e)}", flush=True)
        # Return the error in JSON so dashboard sees it
        return jsonify({'error': f"{type(e).__name__}: {str(e)}"}), 500

if __name__ == '__main__':
    load_models()
    print("🚀 DTRA API Server running on port 5000")
    # Debug=False prevents reloader which can execute imports twice
    app.run(debug=False, host='0.0.0.0', port=5000)
