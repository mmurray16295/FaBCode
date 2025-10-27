import json

weights = json.load(open(r'c:\VS Code\FaB Code\data\card_weights_all_printings.json', 'r', encoding='utf-8'))

cc_heroes = weights['formats']['cc']
print(f'Total CC heroes in weights: {len(cc_heroes)}\n')

young_in_cc = [(name, data['deck_count'], data.get('is_young', False)) for name, data in cc_heroes.items() if data.get('is_young', False)]
adult_in_cc = [(name, data['deck_count'], data.get('is_young', True)) for name, data in cc_heroes.items() if not data.get('is_young', False)]

print(f'Young heroes in CC weighted list: {len(young_in_cc)}')
print(f'Adult heroes in CC weighted list: {len(adult_in_cc)}\n')

if young_in_cc:
    print('Top 20 Young heroes in CC (by deck count):')
    for name, count, _ in sorted(young_in_cc, key=lambda x: x[1], reverse=True)[:20]:
        print(f'  • {name} - {count} decks')
