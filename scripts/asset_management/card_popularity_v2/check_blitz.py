import json

with open('c:/VS Code/FaB Code/data/card.json') as f:
    cards = json.load(f)

heroes = [c for c in cards if 'Hero' in c.get('types', [])]
young_heroes = [h for h in heroes if 'Young' in h.get('types', [])]

young_blitz_legal = [h for h in young_heroes if h.get('blitz_legal')]
young_blitz_illegal = [h for h in young_heroes if not h.get('blitz_legal')]

print(f'Total young heroes: {len(young_heroes)}')
print(f'Young heroes blitz legal: {len(young_blitz_legal)}')
print(f'Young heroes NOT blitz legal: {len(young_blitz_illegal)}')

if young_blitz_illegal:
    print(f'\n❌ Young heroes NOT blitz legal:')
    for h in young_blitz_illegal:
        set_id = h['printings'][0]['set_id'] if h['printings'] else 'Unknown'
        cc = '✓' if h.get('cc_legal') else '✗'
        ll = '✓' if h.get('ll_legal') else '✗'
        commoner = '✓' if h.get('commoner_legal') else '✗'
        print(f'  - {h["name"]}')
        print(f'      Set: {set_id} | CC: {cc} | LL: {ll} | Commoner: {commoner}')
        print(f'      Keywords: {h.get("card_keywords", [])}')
        print()
