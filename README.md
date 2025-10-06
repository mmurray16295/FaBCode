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

## Persistent classes and multi-set generation
The synthetic generator maintains a persistent global class index at:

- `data/synthetic 2/classes.yaml` (same folder as your OUTPUT_BASE_DIR)

Each run:
- Loads the existing class list (`names` array)
- Scans `--card-dirs` for PNGs and appends any new card names
- Reuses existing IDs for previously seen names (handles reprints)
- Writes labels using stable IDs and updates `data.yaml` with the union of all names

### Examples
- Generate 20 images from SEA only:

```sh
python scripts/generate_synthetic_playmat_screenshots.py --num-images 20 --card-dirs "data/images/SEA"
```

- Add WTR later and generate 200 more, preserving IDs:

```sh
python scripts/generate_synthetic_playmat_screenshots.py --num-images 200 --card-dirs "data/images/SEA" "data/images/WTR"
```

- Backgrounds are sampled only from their respective split folders under `data/images/YouTube_Labeled/{train,valid,test}`.

If you want reprints to be treated as different classes, we can switch to prefixing class names with the set directory (e.g., `SEA/CardName` vs `WTR/CardName`).

## Contributing
Pull requests and suggestions are welcome. Please follow standard Python style and document your code.

## License
Specify your license here.
# FaBCode
FaB Card Image Detection and Classification
