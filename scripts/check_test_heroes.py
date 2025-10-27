import json

cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = [
    "Maxx 'The Hype' Nitro",
    "Prism, Sculptor of Arc Light",
    "Fang, Dracai of Blades",
    "Bravo, Star of the Show",
    "Tuffnut, Bumbling Hulkster",
    "Levia, Shadowborn Abomination",
    "Betsy, Skin in the Game",
    "Teklovossen, Esteemed Magnate",
    "Arakni",
    "Briar",
    "Azalea, Ace in the Hole",
    "Enigma, Ledger of Ancestry",
    "Fai, Rising Rebellion",
    "Pleiades, Superstar",
    "Vynnset, Iron Maiden",
    "Oldhim, Grandfather of Eternity",
    "Marlynn, Treasure Hunter",
    "Riptide, Lurker of the Deep",
    "Verdance, Thorn of the Rose",
    "Iyslander, Stormbind",
    "Briar, Warden of Thorns",
    "Dorinthea Ironsong",
    "Zen, Tamer of Purpose",
    "Bravo",
    "Boltyn",
    "Aurora",
    "Ira, Crimson Haze",
    "Fang",
    "Tuffnut",
    "Prism",
    "Puffin",
    "Gravy Bones, Shipwrecked Looter",
    "Kano, Dracai of Aether",
    "Chane"
]

young_count = 0
adult_count = 0

print("=" * 60)
print("HERO AGE BREAKDOWN")
print("=" * 60)

print("\n🟡 YOUNG HEROES:")
for hero_name in test_heroes:
    matching = [c for c in cards if c['name'].lower() == hero_name.lower() or 
                (hero_name.lower().replace("'", "") in c['name'].lower().replace("'", ""))]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        if is_young:
            young_count += 1
            print(f"  • {card['name']}")

print("\n🟢 ADULT HEROES:")
for hero_name in test_heroes:
    matching = [c for c in cards if c['name'].lower() == hero_name.lower() or 
                (hero_name.lower().replace("'", "") in c['name'].lower().replace("'", ""))]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        if not is_young:
            adult_count += 1
            print(f"  • {card['name']}")

print("\n" + "=" * 60)
print(f"SUMMARY:")
print(f"  🟡 Young: {young_count} ({young_count/len(test_heroes)*100:.1f}%)")
print(f"  🟢 Adult: {adult_count} ({adult_count/len(test_heroes)*100:.1f}%)")
print("=" * 60)
