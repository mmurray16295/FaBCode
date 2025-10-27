import json

cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = [
    "Kayo, Berserker Runt",
    "Emperor, Dracai of Aesir",
    "Yoji, Royal Protector",
    "Dash, Database"
]

print("Checking these specific heroes:\n")

for hero_name in test_heroes:
    matching = [c for c in cards if c['name'] == hero_name]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        status = "🟡 YOUNG" if is_young else "🟢 ADULT"
        print(f"{status} - {card['name']}")
        print(f"  Types: {card.get('types', [])}\n")
    else:
        print(f"❌ NOT FOUND - {hero_name}\n")
