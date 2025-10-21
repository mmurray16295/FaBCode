# FaB Card Detector

AI-powered real-time detection and classification of Flesh and Blood trading cards using YOLOv11.

## Features
- **Real-time Card Detection** - Live detection with overlay UI for streaming
- **2,641 Card Classes** - Detects all unique card names across all printings
- **Synthetic Data Generation** - Automated playmat screenshot generation with realistic augmentations
- **Hero-Aware Selection** - Intelligent card selection based on format legality and hero compatibility
- **Popularity Weighting** - Card selection weighted by competitive play data
- **Production Ready** - Packaged Windows application with GUI

## Quick Start

### Prerequisites
- Python 3.8+
- CUDA-compatible GPU (recommended for training)
- 8GB RAM minimum

### Installation
```bash
# Clone repository
git clone https://github.com/mmurray16295/FaBCode.git
cd FaBCode

# Install dependencies
pip install -r requirements.txt

# Download card database
python scripts/asset_management/download_card_json.py
```

### Run Live Detection
```bash
# GUI Application
python scripts/computer_vision_application/fab_detector_app.py

# CLI Tool
python scripts/computer_vision_application/live_detector.py --source screen
```

## Project Structure

```
FaBCode/
├── scripts/
│   ├── asset_management/        # Card data, images, popularity scraping
│   │   └── card_popularity_v2/  # V2 popularity system (CC + Blitz)
│   ├── synthetic_generation/    # Playmat image generation
│   ├── yolo_training/           # Model training scripts
│   ├── class_management/        # Class assignment utilities
│   ├── computer_vision_application/  # Detection apps & packaging
│   └── aws/                     # Cloud infrastructure scripts
├── data/
│   ├── card.json                # Complete card database
│   ├── card_weights_all_printings.json  # Popularity weights
│   └── synthetic/               # Generated training data
└── runs/                        # Training outputs & checkpoints
```

## Workflow

### 1. Generate Training Data
```bash
# Generate 10,000 synthetic playmat images
python scripts/synthetic_generation/Core_Playmat_Generator.py

# Parallel generation (faster, uses multiple cores)
python scripts/synthetic_generation/parallel_generate_dataset.py --num-images 10000
```

### 2. Train Model
```bash
# Full training (300 epochs, YOLOv11x)
python scripts/yolo_training/train_full_yolo11x.py

# Custom training
python scripts/yolo_training/train_yolo11.py \
  --data data/synthetic/data.yaml \
  --weights scripts/yolo_training/yolo11x.pt \
  --epochs 100 \
  --batch 16
```

### 3. Run Detection
```bash
# GUI application with overlay mode
python scripts/computer_vision_application/fab_detector_app.py

# CLI live detection
python scripts/computer_vision_application/live_detector.py --source screen --conf 0.4
```

## Advanced Features

### Card Popularity System V2
Automatically scrapes competitive deck data and weights card selection:
```bash
cd scripts/asset_management/card_popularity_v2
python scrape_popularity.py --formats cc blitz
```

### Custom Dataset Generation
```bash
# Generate with specific augmentation preset
python scripts/synthetic_generation/Core_Playmat_Generator.py --preset phase2

# Test with visualization
python scripts/synthetic_generation/test_generation.py --count 5 --visualize
```

### AWS Training
```bash
# Upload to S3
bash scripts/aws/backup_to_s3.sh

# Download results
python scripts/aws/smart_aws_downloader.py
```

## Documentation

- **[RUNPOD_SETUP.md](RUNPOD_SETUP.md)** - Cloud training setup guide
- **[TROUBLESHOOTING.md](TROUBLESHOOTING.md)** - Common issues and solutions
- **[scripts/synthetic_generation/README.md](scripts/synthetic_generation/README.md)** - Data generation details
- **[scripts/computer_vision_application/README.md](scripts/computer_vision_application/README.md)** - Application deployment
- **[scripts/yolo_training/TRAINING_WORKFLOW.md](scripts/yolo_training/TRAINING_WORKFLOW.md)** - Training best practices

## Performance

- **Training Speed**: ~0.96s per synthetic image (3,750 images/hour)
- **Detection Speed**: 30+ FPS on RTX 3080
- **Model Accuracy**: mAP50 > 0.85 on validation set
- **Dataset Size**: 87,441 training images (train12)

## Contributing
Pull requests and suggestions are welcome. Please follow standard Python style and document your code.

## License
MIT License - See LICENSE file for details
