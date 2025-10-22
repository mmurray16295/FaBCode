import json

# Load the mapping
with open('data/card_name_to_class_id.json', 'r', encoding='utf-8') as f:
    mapping = json.load(f)

# Check special character cards
special_cards = [
    'Jarl Vetreiði',
    'Potion of Déjà Vu', 
    'Riches of Trōpal-Dhani',
    'Tremor of íArathael',
    'Twelve Petal Kāṣāya'
]

print("Class IDs for special character cards:")
for name in special_cards:
    class_id = mapping.get(name, "NOT FOUND")
    print(f"  {name}: class {class_id}")
