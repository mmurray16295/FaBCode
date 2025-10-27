import json

weights = json.load(open(r'c:\VS Code\FaB Code\data\card_weights_all_printings.json', 'r', encoding='utf-8'))
cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = ['Bravo', 'Boltyn', 'Aurora', 'Ira, Crimson Haze', 'Fang', 'Tuffnut', 'Prism', 'Puffin', 'Briar', 'Chane']

print("Checking specific heroes mentioned:\n")

for hero_name in test_heroes:
    # Check in card.json
    card_match = [c for c in cards if c['name'] == hero_name or c['name'].startswith(hero_name)]
    if card_match:
        is_young_card = 'Young' in card_match[0].get('types', [])
        print(f"\n{hero_name}:")
        print(f"  card.json: {'YOUNG' if is_young_card else 'ADULT'} (types: {card_match[0].get('types', [])})")
        
        # Check in CC weights
        if hero_name in weights['formats'].get('cc', {}):
            is_young_weight = weights['formats']['cc'][hero_name].get('is_young', 'NOT SET')
            print(f"  CC weights: is_young = {is_young_weight}")
        else:
            print(f"  CC weights: NOT FOUND")
            
        # Check in Blitz weights
        if hero_name in weights['formats'].get('blitz', {}):
            is_young_weight = weights['formats']['blitz'][hero_name].get('is_young', 'NOT SET')
            print(f"  Blitz weights: is_young = {is_young_weight}")
        else:
            print(f"  Blitz weights: NOT FOUND")
