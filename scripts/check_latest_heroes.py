import json

cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = [
    "Enigma, New Moon",
    "Dromai",
    "Terra",
    "Kano, Dracai of Aether",
    "Gravy Bones",
    "Teklovossen",
    "Dash, Database",
    "Shiyana, Diamond Gemini",
    "Katsu",
    "Kano",
    "Zen, Tamer of Purpose",
    "Brevant, Civic Protector",
    "Pleiades",
    "Arakni",
    "Prism"
]

print("Checking potentially young heroes:\n")

young_count = 0
adult_count = 0

for hero_name in test_heroes:
    matching = [c for c in cards if c['name'] == hero_name or c['name'].startswith(hero_name + ',')]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        status = "🟡 YOUNG" if is_young else "🟢 ADULT"
        print(f"{status} - {card['name']}")
        if is_young:
            young_count += 1
        else:
            adult_count += 1

print(f"\n🟡 Young: {young_count}")
print(f"🟢 Adult: {adult_count}")
