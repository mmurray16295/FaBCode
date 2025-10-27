import json

weights = json.load(open(r'c:\VS Code\FaB Code\data\card_weights_all_printings.json', 'r', encoding='utf-8'))
cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

# Build hero map from card.json
hero_map = {}
for card in cards:
    if 'Hero' in card.get('types', []):
        is_young = 'Young' in card.get('types', [])
        hero_map[card['name']] = is_young

print("Checking Dromai variants in card.json:")
for name, is_young in sorted(hero_map.items()):
    if 'dromai' in name.lower():
        print(f"  {name}: {'YOUNG' if is_young else 'ADULT'}")

print("\nChecking what's in CC weights:")
cc_heroes = weights['formats']['cc']
for name in sorted(cc_heroes.keys()):
    if 'dromai' in name.lower():
        print(f"  {name}")

print("\nThe issue:")
print("  - Weights have: 'Dromai, Ash Artist' (adult version)")
print("  - card.json has: 'Dromai' (young) AND 'Dromai, Ash Artist' (adult)")
print("  - Fuzzy match was matching 'Dromai, Ash Artist' to 'Dromai' (wrong!)")
print("\nSolution: Use EXACT name match only, no fuzzy matching!")
