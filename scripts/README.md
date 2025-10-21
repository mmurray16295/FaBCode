# FaB Card Detector - Scripts Directory

Organized collection of scripts for the Flesh and Blood Card Detector project.

## Directory Structure

```
scripts/
├── README.md                          # This file
├── asset_management/                 # Card data and image management
│   ├── download_card_json.py
│   ├── download_all_printings_parallel.py
│   ├── extract_occluders_from_assets.py
│   └── card_popularity_v2/           # Multi-format card popularity scraping
├── synthetic_generation/             # Synthetic dataset generation
│   ├── Core_Playmat_Generator.py
│   ├── card_selector.py
│   └── augmentation_config.py
├── yolo_training/                    # YOLO model training scripts
│   ├── train_yolo.py
│   └── (training utilities)
├── class_management/                 # Class mapping and organization
│   └── (class management utilities)
└── computer_vision_application/      # Application packaging and deployment
    ├── screen_detect.py              # Main detector application
    ├── build_executable.py
    └── create_windows_package.py
```

## Quick Reference

### Getting Started
```bash
# 1. Download card database
cd scripts/asset_management
python download_card_json.py

# 2. Download card images
python download_all_printings_parallel.py

# 3. Update card popularity data
cd card_popularity_v2
python scrape_popularity.py

# 4. Test synthetic generation
cd ../../synthetic_generation
python Core_Playmat_Generator.py --num-images 1
```

### Common Tasks

**Generate Synthetic Dataset:**
```bash
cd scripts/synthetic_generation
python Core_Playmat_Generator.py --num-images 1000
```

**Train YOLO Model:**
```bash
cd scripts/yolo_training
python train_yolo.py --data data/synthetic/data.yaml --epochs 100
```

**Package Application:**
```bash
cd scripts/computer_vision_application
python create_windows_package.py
```

## Subfolders Overview

### 📦 asset_management/
Manages card data downloads, image processing, and popularity scraping.
- Download card database and images
- Scrape card popularity from fabrec.gg
- Process occluder assets
- See: [asset_management/README.md](asset_management/README.md)

### 🎨 synthetic_generation/
Creates synthetic playmat images for training.
- Generate realistic playmat screenshots
- Apply augmentations (blur, glare, color shifts)
- Card selection based on popularity weights
- See: [synthetic_generation/README.md](synthetic_generation/README.md)

### 🧠 yolo_training/
Training scripts for YOLO object detection models.
- Train YOLO11 models on synthetic data
- Hyperparameter tuning
- Multi-phase training workflows
- See: [yolo_training/README.md](yolo_training/README.md)

### 🔖 class_management/
Manages YOLO class mappings and organization.
- Create class ID mappings
- Handle class hierarchy
- Update class definitions

### 🖥️ computer_vision_application/
Application packaging and deployment.
- Real-time card detection with overlay UI
- Build standalone executables
- Create portable Windows packages
- See: [computer_vision_application/README.md](computer_vision_application/README.md)

## Development Workflow

### Phase 1: Data Preparation
```bash
# Download card data
cd scripts/asset_management
python download_card_json.py
python download_all_printings_parallel.py

# Update popularity weights
cd card_popularity_v2
python scrape_popularity.py
```

### Phase 2: Dataset Generation
```bash
# Test generation (single image)
cd scripts/synthetic_generation
python Core_Playmat_Generator.py --num-images 1

# Full generation
python Core_Playmat_Generator.py --num-images 10000
```

### Phase 3: Training
```bash
# Train model
cd scripts/yolo_training
python train_yolo.py --data data/synthetic/data.yaml

# Monitor training
tensorboard --logdir runs/train
```

### Phase 4: Deployment
```bash
# Package application
cd scripts/computer_vision_application
python create_windows_package.py

# Test detector
python screen_detect.py --weights runs/train/best.pt
```

## Environment Setup

**Python Environment:**
```bash
# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
venv\Scripts\activate     # Windows

# Install dependencies
pip install -r requirements.txt
```

**Required Packages:**
- ultralytics (YOLO)
- opencv-python
- Pillow
- PyYAML
- requests
- beautifulsoup4
- mss (for screen capture)

## Notes

- **Windows Users**: Use Git Bash or WSL for .sh scripts
- **RunPod Environment**: Scripts detect Linux and adjust paths automatically
- **Large Datasets**: Use analysis tools to verify quality before training
- **Model Weights**: Stored in `runs/train/*/weights/`
- **Generated Data**: Saved to `data/synthetic/` by default

## Documentation

Each subfolder contains its own README with detailed documentation:
- [asset_management/README.md](asset_management/README.md)
- [asset_management/card_popularity_v2/README.md](asset_management/card_popularity_v2/README.md)
- [synthetic_generation/README.md](synthetic_generation/README.md)
- [computer_vision_application/README.md](computer_vision_application/README.md)
