# -*- coding: utf-8 -*-
"""
DTRA Configuration Module
Centralized settings and feature definitions for the Dynamic Threat Response Agent.
"""

import os

# --- Project Paths ---
# --- Project Paths ---
BASE_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__))) # Go up one level to root
DATA_DIR = os.path.join(BASE_DIR, "CICIDS2017")
MODELS_DIR = os.path.join(BASE_DIR, "models")

# Create models directory if it doesn't exist
os.makedirs(MODELS_DIR, exist_ok=True)

# --- Model File Paths ---
DETECTOR_MODEL_PATH = os.path.join(MODELS_DIR, "dtra_detector_model.h5")
IMPUTER_PATH = os.path.join(MODELS_DIR, "dtra_imputer.pkl")
SCALER_PATH = os.path.join(MODELS_DIR, "dtra_scaler.pkl")
Q_TABLE_PATH = os.path.join(MODELS_DIR, "dtra_q_table.npy")

# --- The 79 Network Traffic Features (CICIDS2017) ---
NUMERIC_FEATURES = [
    'Destination Port', 'Flow Duration', 'Total Fwd Packets', 'Total Backward Packets',
    'Total Length of Fwd Packets', 'Total Length of Bwd Packets', 'Fwd Packet Length Max',
    'Fwd Packet Length Min', 'Fwd Packet Length Mean', 'Fwd Packet Length Std',
    'Bwd Packet Length Max', 'Bwd Packet Length Min', 'Bwd Packet Length Mean',
    'Bwd Packet Length Std', 'Flow Bytes/s', 'Flow Packets/s', 'Flow IAT Mean',
    'Flow IAT Std', 'Flow IAT Max', 'Flow IAT Min', 'Fwd IAT Total', 'Fwd IAT Mean',
    'Fwd IAT Std', 'Fwd IAT Max', 'Fwd IAT Min', 'Bwd IAT Total', 'Bwd IAT Mean',
    'Bwd IAT Std', 'Bwd IAT Max', 'Bwd IAT Min', 'Fwd PSH Flags', 'Bwd PSH Flags',
    'Fwd URG Flags', 'Bwd URG Flags', 'Fwd Header Length', 'Bwd Header Length',
    'Fwd Packets/s', 'Bwd Packets/s', 'Min Packet Length', 'Max Packet Length',
    'Packet Length Mean', 'Packet Length Std', 'Packet Length Variance',
    'FIN Flag Count', 'SYN Flag Count', 'RST Flag Count', 'PSH Flag Count',
    'ACK Flag Count', 'URG Flag Count', 'CWE Flag Count', 'ECE Flag Count',
    'Down/Up Ratio', 'Average Packet Size', 'Avg Fwd Segment Size',
    'Avg Bwd Segment Size', 'Fwd Header Length.1', 'Fwd Avg Bytes/Bulk',
    'Fwd Avg Packets/Bulk', 'Fwd Avg Bulk Rate', 'Bwd Avg Bytes/Bulk',
    'Bwd Avg Packets/Bulk', 'Bwd Avg Bulk Rate', 'Subflow Fwd Packets',
    'Subflow Fwd Bytes', 'Subflow Bwd Packets', 'Subflow Bwd Bytes',
    'Init_Win_bytes_forward', 'Init_Win_bytes_backward', 'act_data_pkt_fwd',
    'min_seg_size_forward', 'Active Mean', 'Active Std', 'Active Max', 'Active Min',
    'Idle Mean', 'Idle Std', 'Idle Max', 'Idle Min'
]

# --- A* Decision Costs ---
BUSINESS_COSTS = {
    'Ignore': 0,    # Free - no action taken
    'Log': 2,       # Very cheap - just record
    'Block': 20,    # Disruptive - blocks traffic
    'Isolate': 80   # Very disruptive - quarantine system
}

# --- Q-Learning Parameters ---
RL_CONFIG = {
    'num_states': 10,           # Danger score discretized into 10 buckets
    'actions': ['Ignore', 'Log', 'Block', 'Isolate'],
    'alpha': 0.1,               # Learning rate
    'gamma': 0.9,               # Discount factor
    'epsilon': 0.1,             # Exploration rate
    'num_episodes': 10000       # Training episodes
}

# --- Dashboard Settings ---
DASHBOARD_CONFIG = {
    'page_title': "DTRA | Enterprise Sentinel",
    'page_icon': "🛡️",
    'layout': "wide"
}
