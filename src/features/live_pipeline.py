"""
Week 4: Live & PCAP Network Traffic Feature Extraction Pipeline
Uses the official cicflowmeter Python package to extract flows from PCAP files
or live network interfaces, aligns features to trained model's expected schema,
and predicts network intrusions in real time.
"""

import os
import sys
import time
import warnings
import joblib
import numpy as np
import pandas as pd
from datetime import datetime

# Import scapy and cicflowmeter
from scapy.sendrecv import sniff, AsyncSniffer
from cicflowmeter.flow_session import FlowSession

# Default rename dictionary mapping cicflowmeter output columns (snake_case)
# to CICIDS2017 standard dataset column names (Title Case)
CICFLOWMETER_RENAME_MAP = {
    'dst_port': 'Destination Port',
    'flow_duration': 'Flow Duration',
    'tot_fwd_pkts': 'Total Fwd Packets',
    'tot_bwd_pkts': 'Total Backward Packets',
    'totlen_fwd_pkts': 'Total Length of Fwd Packets',
    'totlen_bwd_pkts': 'Total Length of Bwd Packets',
    'fwd_pkt_len_max': 'Fwd Packet Length Max',
    'fwd_pkt_len_min': 'Fwd Packet Length Min',
    'fwd_pkt_len_mean': 'Fwd Packet Length Mean',
    'fwd_pkt_len_std': 'Fwd Packet Length Std',
    'bwd_pkt_len_max': 'Bwd Packet Length Max',
    'bwd_pkt_len_min': 'Bwd Packet Length Min',
    'bwd_pkt_len_mean': 'Bwd Packet Length Mean',
    'bwd_pkt_len_std': 'Bwd Packet Length Std',
    'flow_byts_s': 'Flow Bytes/s',
    'flow_pkts_s': 'Flow Packets/s',
    'flow_iat_mean': 'Flow IAT Mean',
    'flow_iat_std': 'Flow IAT Std',
    'flow_iat_max': 'Flow IAT Max',
    'flow_iat_min': 'Flow IAT Min',
    'fwd_iat_tot': 'Fwd IAT Total',
    'fwd_iat_mean': 'Fwd IAT Mean',
    'fwd_iat_std': 'Fwd IAT Std',
    'fwd_iat_max': 'Fwd IAT Max',
    'fwd_iat_min': 'Fwd IAT Min',
    'bwd_iat_tot': 'Bwd IAT Total',
    'bwd_iat_mean': 'Bwd IAT Mean',
    'bwd_iat_std': 'Bwd IAT Std',
    'bwd_iat_max': 'Bwd IAT Max',
    'bwd_iat_min': 'Bwd IAT Min',
    'fwd_psh_flags': 'Fwd PSH Flags',
    'bwd_psh_flags': 'Bwd PSH Flags',
    'fwd_urg_flags': 'Fwd URG Flags',
    'bwd_urg_flags': 'Bwd URG Flags',
    'fwd_header_len': 'Fwd Header Length',
    'bwd_header_len': 'Bwd Header Length',
    'fwd_pkts_s': 'Fwd Packets/s',
    'bwd_pkts_s': 'Bwd Packets/s',
    'pkt_len_min': 'Min Packet Length',
    'pkt_len_max': 'Max Packet Length',
    'pkt_len_mean': 'Packet Length Mean',
    'pkt_len_std': 'Packet Length Std',
    'pkt_len_var': 'Packet Length Variance',
    'fin_flag_cnt': 'FIN Flag Count',
    'syn_flag_cnt': 'SYN Flag Count',
    'rst_flag_cnt': 'RST Flag Count',
    'psh_flag_cnt': 'PSH Flag Count',
    'ack_flag_cnt': 'ACK Flag Count',
    'urg_flag_cnt': 'URG Flag Count',
    'cwr_flag_count': 'CWE Flag Count',
    'ece_flag_cnt': 'ECE Flag Count',
    'down_up_ratio': 'Down/Up Ratio',
    'pkt_size_avg': 'Average Packet Size',
    'fwd_seg_size_avg': 'Avg Fwd Segment Size',
    'bwd_seg_size_avg': 'Avg Bwd Segment Size',
    'fwd_byts_b_avg': 'Fwd Avg Bytes/Bulk',
    'fwd_pkts_b_avg': 'Fwd Avg Packets/Bulk',
    'fwd_blk_rate_avg': 'Fwd Avg Bulk Rate',
    'bwd_byts_b_avg': 'Bwd Avg Bytes/Bulk',
    'bwd_pkts_b_avg': 'Bwd Avg Packets/Bulk',
    'bwd_blk_rate_avg': 'Bwd Avg Bulk Rate',
    'subflow_fwd_pkts': 'Subflow Fwd Packets',
    'subflow_fwd_byts': 'Subflow Fwd Bytes',
    'subflow_bwd_pkts': 'Subflow Bwd Packets',
    'subflow_bwd_byts': 'Subflow Bwd Bytes',
    'init_fwd_win_byts': 'Init_Win_bytes_forward',
    'init_bwd_win_byts': 'Init_Win_bytes_backward',
    'fwd_act_data_pkts': 'act_data_pkt_fwd',
    'fwd_seg_size_min': 'min_seg_size_forward',
    'active_mean': 'Active Mean',
    'active_std': 'Active Std',
    'active_max': 'Active Max',
    'active_min': 'Active Min',
    'idle_mean': 'Idle Mean',
    'idle_std': 'Idle Std',
    'idle_max': 'Idle Max',
    'idle_min': 'Idle Min',
}

def extract_flows_from_pcap(pcap_path, output_csv=None):
    """
    Extract network flow features from a .pcap file using cicflowmeter FlowSession.
    Returns Pandas DataFrame of extracted raw flows.
    """
    if not os.path.exists(pcap_path):
        raise FileNotFoundError(f"PCAP file not found: {pcap_path}")

    if output_csv is None:
        os.makedirs("data/live", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"data/live/flows_{timestamp}.csv"

    # Ensure parent directory exists
    os.makedirs(os.path.dirname(output_csv), exist_ok=True)

    # Instantiate cicflowmeter FlowSession
    session = FlowSession(output_mode="csv", output=output_csv)

    # Use scapy sniff in offline mode without extra BPF filters to avoid tcpdump dependencies
    sniff(offline=pcap_path, prn=session.process, store=False)
    session.flush_flows()

    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        raise ValueError(f"No flows were generated from PCAP file: {pcap_path}")

    flow_df = pd.read_csv(output_csv)
    print(f"[+] Extracted {len(flow_df)} flows from PCAP: {pcap_path}")
    return flow_df, output_csv


def extract_flows_live(interface, output_csv=None, packet_count=100, timeout=10):
    """
    Capture live packets from network interface and extract flows using cicflowmeter.
    Returns Pandas DataFrame of extracted flows.
    """
    if output_csv is None:
        os.makedirs("data/live", exist_ok=True)
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        output_csv = f"data/live/live_flows_{timestamp}.csv"

    os.makedirs(os.path.dirname(output_csv), exist_ok=True)
    session = FlowSession(output_mode="csv", output=output_csv)

    print(f"[*] Starting live packet capture on interface: {interface} (count={packet_count}, timeout={timeout}s)...")
    try:
        sniff(iface=interface, prn=session.process, count=packet_count, timeout=timeout, store=False)
    except Exception as e:
        print(f"[!] Warning during live capture: {e}")
        print("    If using Windows, make sure Npcap is installed in WinPcap API-compatible Mode.")
    
    session.flush_flows()

    if not os.path.exists(output_csv) or os.path.getsize(output_csv) == 0:
        print(f"[!] No flows were captured on interface {interface} during the timeout window.")
        return pd.DataFrame(), output_csv

    flow_df = pd.read_csv(output_csv)
    print(f"[+] Extracted {len(flow_df)} live flows from interface: {interface}")
    return flow_df, output_csv


def align_and_clean_features(flow_df, model):
    """
    Aligns raw cicflowmeter output DataFrame with model's expected 78 feature columns.
    
    1. Reads model.feature_names_in_ dynamically.
    2. Applies explicit column rename dictionary.
    3. Handles duplicated 'Fwd Header Length.1' column.
    4. Checks for missing expected features, prints warnings, and fills with 0.
    5. Replaces +Inf / -Inf with NaN, then fills NaN with 0.
    6. Reorders columns to match model.feature_names_in_ exactly.
    
    Returns: Cleaned DataFrame with shape (N, 78) ready for model.predict().
    """
    if flow_df.empty:
        raise ValueError("Input flow DataFrame is empty.")

    # 1. Dynamically retrieve expected feature names from model
    expected_features = list(model.feature_names_in_)
    print(f"[*] Model expects {len(expected_features)} feature columns.")

    # 2. Rename columns using explicit mapping
    df = flow_df.rename(columns=CICFLOWMETER_RENAME_MAP).copy()

    # 3. Handle duplicated 'Fwd Header Length.1' column if missing
    if 'Fwd Header Length.1' in expected_features and 'Fwd Header Length.1' not in df.columns:
        if 'Fwd Header Length' in df.columns:
            df['Fwd Header Length.1'] = df['Fwd Header Length']

    # 4. Check for missing expected features
    missing_cols = [c for c in expected_features if c not in df.columns]
    if missing_cols:
        print(f"[!] WARNING: {len(missing_cols)} expected feature columns were missing in cicflowmeter output!")
        print(f"    Missing columns: {missing_cols}")
        print("    Filling missing feature columns with default values...")
        for c in missing_cols:
            if 'Init_Win' in c:
                df[c] = -1.0
            else:
                df[c] = 0.0

    # 5. Filter to expected features only and clean Inf / NaN
    X = df[expected_features].copy()

    # Replace Inf with NaN, then fill NaN with 0 (matching Week 2 cleaning logic)
    X = X.replace([np.inf, -np.inf], np.nan)
    num_nans = X.isna().sum().sum()
    if num_nans > 0:
        print(f"[*] Replaced {num_nans} Inf/NaN values with 0.")
        X = X.fillna(0.0)

    # 6. Scale Time Columns (Seconds -> Microseconds x1e6)
    # cicflowmeter measures duration & IATs in seconds, whereas CICIDS2017 dataset features
    # (Flow Duration, Flow IAT, Fwd IAT, Bwd IAT, Active, Idle) are in microseconds (x1e6).
    time_cols = [c for c in expected_features if any(k in c for k in ['Duration', 'IAT', 'Active', 'Idle'])]
    for tc in time_cols:
        if tc in X.columns:
            X[tc] = X[tc] * 1000000.0

    # 7. Ensure exact ordering and dtype coercion to float64
    X = X[expected_features].astype(np.float64)
    
    non_zero_count = (X != 0).any(axis=0).sum()
    pct_non_zero = (non_zero_count / len(expected_features)) * 100.0
    print(f"[+] Feature matrix aligned successfully. Final shape: {X.shape}")
    print(f"[*] Non-zero feature columns across flows: {non_zero_count} / {len(expected_features)} ({pct_non_zero:.1f}% populated)")
    return X


def predict_intrusions(
    input_source,
    is_live=False,
    interface="Wi-Fi",
    packet_count=100,
    timeout=10,
    model_path="models/final_model.joblib",
    label_encoder_path="models/label_encoder.joblib",
    output_dir="data/live"
):
    """
    End-to-end pipeline:
    1. Loads trained model and label encoder.
    2. Captures flows from PCAP or Live interface.
    3. Aligns features to exact model schema (78 features).
    4. Predicts intrusion classes and class probabilities.
    5. Saves full output (metadata + predictions) to CSV.
    """
    # 1. Load model and label encoder
    if not os.path.exists(model_path) or not os.path.exists(label_encoder_path):
        raise FileNotFoundError(f"Model files not found. Ensure {model_path} and {label_encoder_path} exist.")

    model = joblib.load(model_path)
    label_encoder = joblib.load(label_encoder_path)
    print(f"[+] Loaded model from {model_path}")
    print(f"[+] Loaded label encoder with {len(label_encoder.classes_)} classes.")

    # 2. Extract flows
    if is_live:
        raw_df, flow_csv = extract_flows_live(interface=interface, packet_count=packet_count, timeout=timeout)
    else:
        raw_df, flow_csv = extract_flows_from_pcap(pcap_path=input_source)

    if raw_df.empty:
        print("[!] No flows to process.")
        return None

    # Preserve metadata columns if available
    metadata_cols = [c for c in ['src_ip', 'dst_ip', 'src_port', 'dst_port', 'protocol', 'timestamp'] if c in raw_df.columns]
    meta_df = raw_df[metadata_cols].copy() if metadata_cols else pd.DataFrame(index=raw_df.index)

    # 3. Align features
    X_aligned = align_and_clean_features(raw_df, model)

    # 4. Predict
    print("[*] Running model inference...")
    pred_indices = model.predict(X_aligned)
    pred_labels = label_encoder.inverse_transform(pred_indices)

    # Calculate prediction confidence probabilities if predict_proba is available
    if hasattr(model, "predict_proba"):
        probs = model.predict_proba(X_aligned)
        confidences = np.max(probs, axis=1)
    else:
        confidences = [1.0] * len(pred_labels)

    # 5. Build results DataFrame
    results_df = meta_df.copy()
    results_df['Predicted_Label'] = pred_labels
    results_df['Confidence'] = confidences

    # Append feature columns to results
    results_df = pd.concat([results_df, X_aligned], axis=1)

    # 6. Save predictions to CSV
    os.makedirs(output_dir, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_file = os.path.join(output_dir, f"live_predictions_{timestamp}.csv")
    results_df.to_csv(out_file, index=False)

    print(f"\n[+] Inference complete! Predictions saved to: {out_file}")
    print("\n--- Summary of Predictions ---")
    print(results_df['Predicted_Label'].value_counts())

    return results_df


if __name__ == "__main__":
    print("Testing pipeline module directly...")
    sample_pcap = "data/live/sample_test.pcap"
    if os.path.exists(sample_pcap):
        predict_intrusions(sample_pcap, is_live=False)
    else:
        print("Sample PCAP not found. Generate one first.")
