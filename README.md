# Spiking Neural Network for Power System Fault Detection and Classification

A PyTorch implementation of Spiking Neural Networks (SNNs) using snnTorch for detecting and classifying electrical faults in power systems.

## Features

- **Fault Detection**: Binary classification model (97-98% accuracy) to detect if a fault is present
- **Fault Classification**: Multi-class model (85-86% accuracy) to classify fault types (G, C, B, A)
- **Spike Encoding**: Hybrid and latency-based spike encoding for temporal signal processing
- **SNN Architecture**: Leaky Integrate-and-Fire (LIF) neurons with surrogate gradient learning

## Project Structure

```
.
├── models/                    # SNN model definitions
│   ├── snn_detection.py      # Binary fault detection model
│   └── snn_classification.py # 4-class fault classification model
├── preprocessing/            # Data pipeline
│   ├── load_data.py          # CSV data loading
│   ├── normalize.py          # MinMax scaling
│   ├── windowing.py          # Temporal windowing
│   ├── encode_spikes.py      # Spike encoding methods
│   ├── pipeline.py           # Detection pipeline
│   └── pipeline_classification.py # Classification pipeline
├── training/                 # Training scripts
│   ├── train_detection.py    # Detection model training
│   └── train_classification.py # Classification model training
├── data/
│   └── raw/
│       ├── detect_dataset.csv    # Detection dataset
│       └── classData.csv         # Classification dataset
└── requirements.txt          # Python dependencies
```

## Installation

1. Create and activate virtual environment:
```bash
python -m venv snn_env
# On Windows:
snn_env\Scripts\activate
# On Linux/Mac:
source snn_env/bin/activate
```

2. Install dependencies:
```bash
pip install -r requirements.txt
```

## Usage

### Train Detection Model
```bash
python -m training.train_detection
```

### Train Classification Model
```bash
python -m training.train_classification
```

## Model Performance

| Model | Task | Accuracy |
|-------|------|----------|
| Detection | Binary fault detection | 97-98% |
| Classification | 4-class fault type | 85-86% |

## Configuration

Key hyperparameters in training scripts:
- **Batch size**: 32
- **Learning rate**: 0.005
- **Epochs**: 500-1000
- **Window size**: 50-100 timesteps
- **SNN decay (beta)**: 0.8-0.9

## Data Preprocessing

Input features (6 channels):
- Ia, Ib, Ic (current measurements)
- Va, Vb, Vc (voltage measurements)

Preprocessing steps:
1. MinMax normalization
2. Temporal windowing
3. Spike encoding (latency-based)
4. Train/test split (80/20)

## Dependencies

- PyTorch
- snnTorch
- NumPy
- Pandas
- scikit-learn

See `requirements.txt` for full list.

## Author

Created for power system fault detection research using Spiking Neural Networks.

## License

MIT License
