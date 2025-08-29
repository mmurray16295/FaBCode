# FaB Card Detector

## Project Overview
Detects and classifies Flesh and Blood cards in live video using computer vision and YOLO.

## Features
- Real-time card detection and classification
- Synthetic data generation
- Model training and evaluation

## Setup Instructions
1. Clone the repository
2. Set up a Python virtual environment (ask chat idkwtf that is lol)
3. Install dependencies (see `requirements.txt`)
4. Download card.json (run download_card_json.py)
4. Download card images for the set you want to train (download_images.py)
Setcode is defined on line 8.
5. Generate Synthetic data - recommended 20 images per card (I.E. a set with 200 cards should be trained on 4000 synthetic images)
6. Train and test the model
7. Run screen_detect.py to live test (requires two monitors)

## Usage
### Download Card Images
```sh
python scripts/download_images.py
```

### Generate Synthetic Data
```sh
python scripts/generate_synthetic_data.py --num-images 100
```

### Train Model
To begin training with YOLOv11:

1. Ensure your synthetic data is generated and `data/synthetic/data.yaml` is present.
2. Use the provided model file `yolo11n.pt` (located in the top-level folder) as your base model.
3. To continue training from previous weights, use `best.pt` from `runs/detect/train12/weights/`.  Or latest train folder.

Example training command:
```sh
python train.py --data data/synthetic/data.yaml --weights runs/detect/train12/weights/best.pt --model yolo11n.pt --epochs 100
```

Adjust the command as needed for your environment and training script. Training outputs will be saved in a new folder under `runs/detect/`.

## Data Structure
- `data/images/` - Card images by set
- `data/synthetic/` - Synthetic images and labels
- `runs/detect/train12/` - Latest training run and model weights
- `scripts/` - Utility scripts for data and model management

## Contributing
Pull requests and suggestions are welcome. Please follow standard Python style and document your code.

## License
Specify your license here.
# FaBCode
FaB Card Image Detection and Classification
