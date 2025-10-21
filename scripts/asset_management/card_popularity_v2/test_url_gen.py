import re

def test_url_generation(hero_name, expected_url):
    # New method with special character mapping
    char_map = {
        'ð': 'd',
        'í': 'i',
        'á': 'a',
        'é': 'e',
        'ó': 'o',
        'ú': 'u',
    }
    
    normalized = hero_name.lower()
    for old_char, new_char in char_map.items():
        normalized = normalized.replace(old_char, new_char)
    normalized = normalized.replace(',', '')
    normalized = normalized.replace("'", '')
    normalized = normalized.replace('!', '')
    normalized = re.sub(r'[^a-z0-9\s-]', '', normalized)
    normalized = re.sub(r'\s+', '-', normalized.strip())
    
    print(f"Hero: {hero_name}")
    print(f"  Generated: {normalized}")
    print(f"  Expected:  {expected_url}")
    print(f"  Match: {'✓' if normalized == expected_url else '✗'}")
    print()

# Test cases
test_url_generation("Boltyn", "boltyn")
test_url_generation("Ser Boltyn, Breaker of Dawn", "ser-boltyn-breaker-of-dawn")
test_url_generation("Jarl Vetreiði", "jarl-vetreidi")
test_url_generation("Dorinthea Ironsong", "dorinthea-ironsong")
