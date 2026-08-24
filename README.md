# Network Traffic Analyzer & Intrusion Detection System

CN-based project: ML-driven intrusion detection, trained offline on CICIDS2017
and deployed against live traffic (or replayed .pcap files) for real-time
classification.

## Setup

```bash
python -m venv venv
venv\Scripts\activate        # Windows
pip install -r requirements.txt
```

Also install **Npcap** (with WinPcap-compatible mode) for live packet capture —
see npcap.com.

## Getting the dataset

Download CICIDS2017 (the pre-extracted flow-feature CSVs — search "CICIDS2017"
on Kaggle, or get it from UNB's official page: unb.ca/cic/datasets/ids-2017.html).
Unzip the CSVs into `data/raw/`.

## Project structure

```
network-ids/
├── data/
│   ├── raw/            # downloaded CICIDS2017 CSVs go here
│   └── processed/      # cleaned/preprocessed data (Week 2)
├── notebooks/
│   └── 01_eda.ipynb    # Week 1: exploratory data analysis
├── src/
│   ├── capture/        # scapy sniffer / pcap reader (Week 4)
│   ├── features/       # flow reconstruction + feature extraction (Week 4)
│   ├── model/          # training + evaluation scripts (Week 2-3)
│   └── dashboard/       # Streamlit/Flask live view (Week 5)
├── models/              # saved trained model (.pkl / .joblib)
└── report/              # final report, plots, confusion matrices
```

## Weekly plan

1. **Setup + Data Understanding** — env, download data, EDA (`notebooks/01_eda.ipynb`)
2. **Preprocessing + Baseline Model** — clean data, handle class imbalance, train Random Forest
3. **Model Improvement + Evaluation** — compare models (XGBoost), tune, feature importance
4. **Feature Extraction Pipeline** — live-capture flow reconstruction matching training schema
5. **Real-Time Detection + Dashboard** — wire model into live inference + Streamlit dashboard
6. **Report, Polish, Viva Prep** — write-up, architecture diagram, limitations, README cleanup

## Running the EDA notebook

```bash
jupyter notebook notebooks/01_eda.ipynb
```

Make sure the CICIDS2017 CSVs are in `data/raw/` before running.
