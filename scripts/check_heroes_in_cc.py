import json

weights = json.load(open(r'c:\VS Code\FaB Code\data\card_weights_all_printings.json', 'r', encoding='utf-8'))

test_heroes = ["Enigma, New Moon", "Dromai", "Terra", "Gravy Bones", "Teklovossen", "Dash, Database", 
               "Shiyana, Diamond Gemini", "Katsu", "Kano", "Brevant, Civic Protector", "Pleiades", "Arakni", "Prism"]

print("Checking if these heroes are in CC weighted list:\n")

cc_heroes = weights['formats']['cc']

for hero_name in test_heroes:
    # Try exact match and partial matches
    found_in_cc = None
    for cc_hero_key in cc_heroes.keys():
        if hero_name.lower() in cc_hero_key.lower() or cc_hero_key.lower() in hero_name.lower():
            found_in_cc = cc_hero_key
            break
    
    if found_in_cc:
        is_young = cc_heroes[found_in_cc].get('is_young', 'NOT SET')
        print(f"✓ {hero_name}")
        print(f"    Found as: '{found_in_cc}'")
        print(f"    is_young: {is_young}")
    else:
        print(f"✗ {hero_name} - NOT IN CC WEIGHTS")
