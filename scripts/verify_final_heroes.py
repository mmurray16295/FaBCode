import json

cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = [
    "Dash, Inventor Extraordinaire",
    "Lyath Goldmane, Vile Savant",
    "Kayo, Berserker Runt",
    "Emperor, Dracai of Aesir",
    "Dorinthea Ironsong",
    "Jarl Vetreiði",
    "Arakni, 5L!p3d 7hRu 7h3 cR4X",
    "Levia, Shadowborn Abomination",
    "Arakni, Marionette",
    "Kayo, Underhanded Cheat",
    "Prism, Sculptor of Arc Light",
    "Prism, Awakener of Sol",
    "Victor Goldmane, High and Mighty",
    "Gravy Bones, Shipwrecked Looter",
    "Kano, Dracai of Aether",
    "Nuu, Alluring Desire",
    "Verdance, Thorn of the Rose",
    "Riptide, Lurker of the Deep",
    "Yoji, Royal Protector",
    "Dash, Database"
]

young_count = 0
adult_count = 0

print("=" * 60)
for hero_name in test_heroes:
    matching = [c for c in cards if c['name'] == hero_name or hero_name.replace('≡', '') in c['name']]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        status = "🟡 YOUNG" if is_young else "🟢 ADULT"
        print(f"{status} - {card['name']}")
        if is_young:
            young_count += 1
        else:
            adult_count += 1
    else:
        print(f"❌ NOT FOUND - {hero_name}")

print("=" * 60)
print(f"🟡 Young: {young_count}")
print(f"🟢 Adult: {adult_count}")
print(f"✅ SUCCESS! All heroes are adults!" if young_count == 0 else f"⚠️  Still have {young_count} young heroes")
