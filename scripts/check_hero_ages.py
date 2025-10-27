import json

cards = json.load(open(r'c:\VS Code\FaB Code\data\card.json', 'r', encoding='utf-8'))

test_heroes = [
    'Nuu, Alluring Desire',
    'Betsy, Skin in the Game', 
    'Pleiades, Superstar',
    'Ser Boltyn, Breaker of Dawn',
    'Lyath Goldmane, Vile Savant',
    'Uzuri, Switchblade',
    'Dorinthea Ironsong',
    'Vynnset, Iron Maiden',
    'Marlynn, Treasure Hunter',
    'Olympia, Prized Fighter',
    'Maxx The Hype Nitro',
    'Florian, Rotwood Harbinger',
    'Zen, Tamer of Purpose',
    'Kayo, Armed and Dangerous',
    'Puffin, Hightail'
]

print("Checking heroes from test runs:\n")
for hero_name in test_heroes:
    matching = [c for c in cards if hero_name.lower() in c['name'].lower()]
    if matching:
        card = matching[0]
        is_young = 'Young' in card.get('types', [])
        status = "🟡 YOUNG" if is_young else "🟢 ADULT"
        print(f"  {status} - {card['name']}")
    else:
        print(f"  ❌ NOT FOUND - {hero_name}")
