#!/usr/bin/env python3
"""
List cards that legitimately CANNOT be selected by the card selector,
excluding dual-class/dual-talent issues (which are now fixed).

These are cards that have fundamental issues preventing selection.
"""

import json
from collections import defaultdict

def has_split_class_or_talent(card_types, hero_classes_exist, hero_talents_exist):
    """
    Check if this card has split class/talent that would have been caught by the OLD bug.
    We want to EXCLUDE these from the "unfixable" list since they're now fixed.
    """
    from analyze_missing_deep import ALL_CLASSES, ALL_TALENTS
    
    card_classes = card_types & ALL_CLASSES
    card_talents = card_types & ALL_TALENTS
    
    # If card has multiple classes or talents, it was a victim of the split-class bug
    if len(card_classes) > 1 or len(card_talents) > 1:
        return True
    
    return False

def main():
    # Load the deep analysis
    with open('DEEP_MISSING_ANALYSIS.json', 'r') as f:
        analysis = json.load(f)
    
    # Load card database
    with open('data/card.json', 'r') as f:
        card_db = json.load(f)
    
    # Import classes/talents
    import sys
    sys.path.insert(0, 'scripts')
    from card_selector import ALL_CLASSES, ALL_TALENTS
    
    # Categorize the never-selectable cards
    categories = {
        'not_legal_format': [],      # Not legal in CC (the main format)
        'hero_cards': [],             # Hero cards themselves
        'talent_mismatch': [],        # Has talents but no hero has those talents
        'class_mismatch': [],         # Has classes but no hero has those classes
        'other': []                   # Other reasons
    }
    
    for card_info in analysis['never_selectable_cards']:
        class_id = card_info['class_id']
        card_name = card_info['name']
        reason = card_info['reason']
        card_types = set(card_info['types'])
        
        # Check if this was a split-class/talent victim (now fixed)
        if has_split_class_or_talent(card_types, True, True):
            continue  # Skip - this is fixed now
        
        # Categorize by reason
        if 'Not legal in cc' in reason:
            categories['not_legal_format'].append(card_info)
        elif 'Is a hero card' in reason:
            categories['hero_cards'].append(card_info)
        elif 'talents' in reason.lower() and 'but hero has no talents' in reason:
            categories['talent_mismatch'].append(card_info)
        elif 'classes' in reason.lower() and "don't match" in reason:
            categories['class_mismatch'].append(card_info)
        else:
            categories['other'].append(card_info)
    
    # Print report
    print("="*80)
    print("CARDS THAT CANNOT BE SELECTED (Excluding Fixed Split-Class Issues)")
    print("="*80)
    
    total = sum(len(cards) for cards in categories.values())
    print(f"\nTotal: {total} unique cards\n")
    print("These cards have fundamental issues preventing selection:")
    print("- Not legal in Classic Constructed format")
    print("- Are hero cards themselves")
    print("- Have talent/class requirements no hero can fulfill")
    print("\n" + "="*80 + "\n")
    
    # Not legal in format
    if categories['not_legal_format']:
        print(f"🔴 NOT LEGAL IN CLASSIC CONSTRUCTED ({len(categories['not_legal_format'])} cards)")
        print("These are Event cards, UPF-only cards, or format-restricted cards\n")
        for i, card in enumerate(sorted(categories['not_legal_format'], key=lambda x: x['name']), 1):
            types_str = ', '.join(card['types'])
            print(f"  {i:3d}. {card['name']:45s} | {types_str}")
        print()
    
    # Hero cards
    if categories['hero_cards']:
        print(f"🟠 HERO CARDS ({len(categories['hero_cards'])} cards)")
        print("These are heroes themselves, correctly excluded from selection\n")
        for i, card in enumerate(sorted(categories['hero_cards'], key=lambda x: x['name']), 1):
            types_str = ', '.join(card['types'])
            print(f"  {i:3d}. {card['name']:45s} | {types_str}")
        print()
    
    # Talent mismatch
    if categories['talent_mismatch']:
        print(f"🟡 TALENT MISMATCH ({len(categories['talent_mismatch'])} cards)")
        print("Cards requiring talents that no hero currently has\n")
        
        # Group by talent
        by_talent = defaultdict(list)
        for card in categories['talent_mismatch']:
            card_types = set(card['types'])
            card_talents = card_types & ALL_TALENTS
            talent_str = ', '.join(sorted(card_talents))
            by_talent[talent_str].append(card)
        
        for talent_str, cards in sorted(by_talent.items()):
            print(f"\n  Talent: {talent_str} ({len(cards)} cards)")
            for card in sorted(cards, key=lambda x: x['name']):
                print(f"    - {card['name']}")
        print()
    
    # Class mismatch
    if categories['class_mismatch']:
        print(f"🟢 CLASS MISMATCH ({len(categories['class_mismatch'])} cards)")
        print("Cards requiring classes that no hero currently has\n")
        
        # Group by class
        by_class = defaultdict(list)
        for card in categories['class_mismatch']:
            card_types = set(card['types'])
            card_classes = card_types & ALL_CLASSES
            class_str = ', '.join(sorted(card_classes))
            by_class[class_str].append(card)
        
        for class_str, cards in sorted(by_class.items()):
            print(f"\n  Class: {class_str} ({len(cards)} cards)")
            for card in sorted(cards, key=lambda x: x['name']):
                print(f"    - {card['name']}")
        print()
    
    # Other
    if categories['other']:
        print(f"🔵 OTHER ISSUES ({len(categories['other'])} cards)")
        print("Cards with other fundamental selection issues\n")
        for i, card in enumerate(sorted(categories['other'], key=lambda x: x['name']), 1):
            types_str = ', '.join(card['types'])
            print(f"  {i:3d}. {card['name']:45s} | {types_str}")
            print(f"       Reason: {card['reason']}")
        print()
    
    # Save report
    report = {
        'summary': {
            'total_unfixable': total,
            'not_legal_format': len(categories['not_legal_format']),
            'hero_cards': len(categories['hero_cards']),
            'talent_mismatch': len(categories['talent_mismatch']),
            'class_mismatch': len(categories['class_mismatch']),
            'other': len(categories['other'])
        },
        'categories': {
            'not_legal_format': [c['name'] for c in categories['not_legal_format']],
            'hero_cards': [c['name'] for c in categories['hero_cards']],
            'talent_mismatch': [c['name'] for c in categories['talent_mismatch']],
            'class_mismatch': [c['name'] for c in categories['class_mismatch']],
            'other': [c['name'] for c in categories['other']]
        },
        'detailed': categories
    }
    
    with open('CARDS_CANNOT_BE_SELECTED.json', 'w') as f:
        json.dump(report, f, indent=2)
    
    print("="*80)
    print("SUMMARY")
    print("="*80)
    print(f"\nNot legal in format:  {len(categories['not_legal_format']):3d} cards")
    print(f"Hero cards:           {len(categories['hero_cards']):3d} cards")
    print(f"Talent mismatch:      {len(categories['talent_mismatch']):3d} cards")
    print(f"Class mismatch:       {len(categories['class_mismatch']):3d} cards")
    print(f"Other issues:         {len(categories['other']):3d} cards")
    print(f"{'─'*40}")
    print(f"Total unfixable:      {total:3d} cards")
    
    print(f"\n✅ Saved detailed report to: CARDS_CANNOT_BE_SELECTED.json")
    print("="*80 + "\n")

if __name__ == '__main__':
    main()
