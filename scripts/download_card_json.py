import os
import requests
import json

RAW_URL = "https://raw.githubusercontent.com/the-fab-cube/flesh-and-blood-cards/develop/json/english/card.json"
LOCAL_PATH = os.path.join(os.path.dirname(__file__), '..', 'data', 'card.json')

def download_json():
    print(f"Downloading latest card data from: {RAW_URL}")
    print(f"Saving to: {LOCAL_PATH}")
    try:
        response = requests.get(RAW_URL)
        response.raise_for_status()
        with open(LOCAL_PATH, 'w', encoding='utf-8') as f:
            f.write(response.text)
        print(f"Card data saved to {LOCAL_PATH}")
    except Exception as e:
        print(f"Error downloading or saving card data: {e}")

def read_local_json():
    with open(LOCAL_PATH, 'r', encoding='utf-8') as f:
        return json.load(f)

if __name__ == "__main__":
    download_json()
    card_data = read_local_json()
    print(f"Loaded {len(card_data)} cards.")
