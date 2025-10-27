import json

weights = json.load(open(r'c:\VS Code\FaB Code\data\card_weights_all_printings.json', 'r', encoding='utf-8'))

cc = weights['formats']['cc']
blitz = weights['formats']['blitz']

heroes = [
    'Kayo, Berserker Runt',
    'Emperor, Dracai of Aesir', 
    'Yoji, Royal Protector',
    'Dash, Database'
]

print("Checking if these young heroes are in weights:\n")

for h in heroes:
    in_cc = h in cc
    in_blitz = h in blitz
    
    if in_cc:
        is_young_flag = cc[h].get('is_young', 'NOT SET')
        print(f"✓ {h}")
        print(f"  IN CC: YES - is_young={is_young_flag}")
    elif in_blitz:
        is_young_flag = blitz[h].get('is_young', 'NOT SET')
        print(f"✓ {h}")
        print(f"  IN CC: NO")
        print(f"  IN BLITZ: YES - is_young={is_young_flag}")
    else:
        print(f"✗ {h} - NOT IN ANY FORMAT")
    print()
