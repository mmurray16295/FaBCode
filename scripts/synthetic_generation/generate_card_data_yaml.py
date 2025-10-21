"""
Generate data.yaml for YOLO training with card names as classes.
Each unique card name becomes a class, regardless of printing/edition.
"""

import json
from pathlib import Path
from collections import OrderedDict

def generate_card_data_yaml():
    """Generate data.yaml with all unique card names as classes."""
    
    # Load card.json
    card_json_path = Path(__file__).parent.parent / 'data' / 'card.json'
    print(f"Loading cards from {card_json_path}")
    
    with open(card_json_path, 'r', encoding='utf-8') as f:
        all_cards = json.load(f)
    
    # Extract unique card names (case-sensitive)
    unique_names = OrderedDict()
    for card in all_cards:
        name = card.get('name')
        if name and name not in unique_names:
            unique_names[name] = True
    
    # Sort alphabetically for consistency
    sorted_names = sorted(unique_names.keys())
    
    print(f"\nFound {len(sorted_names)} unique card names")
    print(f"First 10: {sorted_names[:10]}")
    print(f"Last 10: {sorted_names[-10:]}")
    
    # Create data.yaml content
    output_path = Path(__file__).parent.parent / 'data' / 'synthetic' / 'data.yaml'
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    yaml_content = f"""# YOLO Dataset Configuration - Card Recognition
# Auto-generated from card.json
# Each class represents a unique Flesh and Blood card name
# Multiple printings/editions of the same card share the same class

train: train/images
val: valid/images
test: test/images

nc: {len(sorted_names)}
names: {sorted_names}
"""
    
    output_path.write_text(yaml_content, encoding='utf-8')
    print(f"\n✓ Created {output_path}")
    print(f"  Classes: {len(sorted_names)}")
    
    # Also create a mapping file for lookup
    mapping_path = Path(__file__).parent.parent / 'data' / 'card_name_to_class_id.json'
    
    card_name_to_id = {name: idx for idx, name in enumerate(sorted_names)}
    
    with open(mapping_path, 'w', encoding='utf-8') as f:
        json.dump(card_name_to_id, f, indent=2, ensure_ascii=False)
    
    print(f"✓ Created {mapping_path}")
    print(f"  Mapping: card_name -> class_id (0-{len(sorted_names)-1})")
    
    return card_name_to_id

if __name__ == '__main__':
    print("=" * 70)
    print("Generating Card-Based data.yaml for YOLO Training")
    print("=" * 70)
    
    mapping = generate_card_data_yaml()
    
    print("\n" + "=" * 70)
    print("Example mappings:")
    print("=" * 70)
    
    # Show some example mappings
    example_cards = list(mapping.items())[:5]
    for name, class_id in example_cards:
        print(f"  {class_id}: {name}")
    
    print("\n✓ Done! Use card_name_to_class_id.json in Core_Playmat_Generator.py")
