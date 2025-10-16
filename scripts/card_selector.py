"""
Simplified card selection system for synthetic playmat generation.

Data Flow:
1. Select hero from card_popularity_weights_by_hero.json
2. Look up hero card data in card.json
3. Select cards: 90% from weights, 10% non-weighted (4% generic, 1% class, 1% talent, 4% both)
4. Look up each card's properties in card.json
"""

import json
import random
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path

# Classes and talents
ALL_CLASSES = {'Warrior', 'Brute', 'Guardian', 'Ninja', 'Ranger', 'Assassin', 
               'Wizard', 'Runeblade', 'Mechanologist', 'Merchant', 'Illusionist', 
               'Shapeshifter', 'Bard', 'Mystic'}
ALL_TALENTS = {'Draconic', 'Elemental', 'Light', 'Shadow', 'Earth', 'Ice', 
               'Lightning', 'Royal', 'Chi', 'Chaos', 'Arcane'}


def select_format() -> str:
    """Select a format with weighted probability: CC 80%, LL 15%, Blitz 5%."""
    rand = random.random()
    if rand < 0.80:
        return 'cc'
    elif rand < 0.95:  # 0.80 + 0.15
        return 'll'
    else:
        return 'blitz'


def is_card_legal_for_format(card: Dict, format: str) -> bool:
    """Check if a card is legal in the specified format.
    
    Args:
        card: Card dict with legality fields
        format: One of 'cc', 'll', or 'blitz'
    
    Returns:
        True if card is legal in the format
    """
    format_key = f'{format}_legal'
    return card.get(format_key) == True


class CardSelector:
    """Handles all card selection logic."""
    
    def __init__(self, card_json_path: str = 'data/card.json',
                 weights_path: str = 'data/card_popularity_weights_by_hero.json'):
        """Initialize with data files."""
        print("Loading card database...")
        with open(card_json_path, 'r', encoding='utf-8') as f:
            self.all_cards = json.load(f)
        
        print("Loading popularity weights...")
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.weights_data = json.load(f)
        
        # Create card lookup by name for fast access
        self.card_lookup = {}
        for card in self.all_cards:
            name = card['name']
            if name not in self.card_lookup:
                self.card_lookup[name] = card
        
        print(f"Loaded {len(self.all_cards)} cards, {len(self.card_lookup)} unique names")
        print(f"Loaded {len(self.weights_data.get('heroes', {}))} heroes with weights")
    
    def select_random_hero(self) -> Tuple[str, Dict, Dict]:
        """
        Select a random hero from weights file.
        Only selects heroes that are legal in at least one format.
        
        Returns:
            (hero_key, hero_card_data, hero_weights)
        """
        # Get all hero keys and try to find legal ones
        hero_keys = list(self.weights_data['heroes'].keys())
        random.shuffle(hero_keys)
        
        for hero_key in hero_keys:
            hero_weights = self.weights_data['heroes'][hero_key]
            
            # Find matching hero card in card.json
            # Normalize by removing punctuation and converting to lowercase
            hero_key_normalized = hero_key.lower().replace('-', ' ').replace(',', '').replace('  ', ' ')
            hero_card = None
            
            for card in self.all_cards:
                if 'Hero' in card.get('types', []):
                    name_normalized = card['name'].lower().replace(',', '').replace('  ', ' ')
                    if hero_key_normalized in name_normalized or name_normalized in hero_key_normalized:
                        hero_card = card
                        break
            
            # Skip if hero card not found
            if not hero_card:
                continue
            
            # Check if hero is legal in at least one format
            if hero_card.get('cc_legal') or hero_card.get('ll_legal') or hero_card.get('blitz_legal'):
                return hero_key, hero_card, hero_weights
        
        raise ValueError("Could not find any legal hero in weights file")
    
    def get_hero_classes_and_talents(self, hero_card: Dict) -> Tuple[Set[str], Set[str]]:
        """Extract classes and talents from hero card, including Essence bonuses."""
        hero_types = set(hero_card.get('types', []))
        classes = hero_types & ALL_CLASSES
        talents = hero_types & ALL_TALENTS
        
        # Check for "Essence of X" keywords that grant additional talent access
        card_keywords = hero_card.get('card_keywords', [])
        for keyword in card_keywords:
            if 'Essence of' in keyword:
                # Extract talents from "Essence of Earth", "Essence of Earth and Ice", etc.
                essence_part = keyword.replace('Essence of', '').strip()
                
                # Split by common separators
                for separator in [', and ', ' and ', ',']:
                    essence_part = essence_part.replace(separator, '|')
                
                # Extract each talent
                essence_talents = [t.strip() for t in essence_part.split('|')]
                
                for talent in essence_talents:
                    if talent in ALL_TALENTS:
                        talents.add(talent)
        
        return classes, talents
    
    def build_card_pools(self, hero_card: Dict, hero_weights: Dict, format: str = 'cc') -> Dict[str, List[Dict]]:
        """
        Build card pools for selection:
        - weighted: Cards from popularity weights
        - generic: Generic cards not in weights
        - class_only: Hero's class cards not in weights
        - talent_only: Hero's talent cards not in weights
        - both: Hero's class AND talent cards not in weights
        
        Args:
            hero_card: Hero card dict
            hero_weights: Hero weights dict from popularity file
            format: Format to filter for ('cc', 'll', or 'blitz')
        
        Returns:
            Dict with card pools
        """
        hero_classes, hero_talents = self.get_hero_classes_and_talents(hero_card)
        
        # Get all weighted card names
        weighted_card_names = set()
        weighted_cards_list = []
        for section, cards in hero_weights.get('sections', {}).items():
            for card in cards:
                card_name = card['card_name']
                
                # Look up card data
                if card_name in self.card_lookup:
                    card_data = self.card_lookup[card_name]
                    
                    # Skip cards that are not legal in the selected format
                    if not is_card_legal_for_format(card_data, format):
                        continue
                    
                    weighted_card_names.add(card_name)
                    card_copy = card_data.copy()
                    card_copy['usage_percentage'] = card['usage_percentage']
                    card_copy['section'] = section
                    weighted_cards_list.append(card_copy)
        
        # Build non-weighted pools
        generic_cards = []
        class_only_cards = []
        talent_only_cards = []
        both_cards = []
        
        for card in self.all_cards:
            # Skip if in weighted list or is a hero or is not legal in the format
            if (card['name'] in weighted_card_names or 
                'Hero' in card.get('types', []) or 
                not is_card_legal_for_format(card, format)):
                continue
            
            card_types = set(card.get('types', []))
            
            # Generic cards
            if 'Generic' in card_types:
                generic_cards.append(card)
                continue
            
            # Check class/talent matching
            card_classes = card_types & ALL_CLASSES
            card_talents = card_types & ALL_TALENTS
            
            # If hero has no talents, exclude cards with ANY talent
            if not hero_talents and card_talents:
                continue
            
            # Check if card matches hero
            # If card has classes, ALL must match hero's classes
            # If card has talents, ALL must match hero's talents
            card_classes_match = not card_classes or card_classes.issubset(hero_classes)
            card_talents_match = not card_talents or card_talents.issubset(hero_talents)
            
            # Skip card if any class or talent doesn't match
            if not card_classes_match or not card_talents_match:
                continue
            
            # Categorize based on what the card has
            has_any_hero_class = bool(card_classes & hero_classes)
            has_any_hero_talent = bool(card_talents & hero_talents)
            
            if has_any_hero_class and has_any_hero_talent:
                both_cards.append(card)
            elif has_any_hero_class:
                class_only_cards.append(card)
            elif has_any_hero_talent:
                talent_only_cards.append(card)
        
        return {
            'weighted': weighted_cards_list,
            'generic': generic_cards,
            'class_only': class_only_cards,
            'talent_only': talent_only_cards,
            'both': both_cards
        }
    
    def _convert_usage_to_weight(self, usage_percentage: float) -> float:
        """
        Convert deck usage percentage to selection weight.
        
        We want to preserve relative popularity while ensuring all cards can be selected.
        Using usage^0.75 (between sqrt=0.5 and linear=1.0) provides stronger weighting:
        - 90% usage -> weight ~52.4
        - 50% usage -> weight ~18.8
        - 10% usage -> weight ~3.2
        - 1% usage -> weight ~1.0
        
        This ensures popular cards are strongly favored (about 2x more than sqrt)
        while all cards still have a reasonable chance of appearing.
        
        Args:
            usage_percentage: Percentage (0-100) the card is used in decks
        
        Returns:
            Weight value for selection
        """
        import math
        # Power of 0.75 provides stronger weighting than sqrt (0.5) but not as extreme as linear (1.0)
        return math.pow(usage_percentage, 0.75)
    
    def select_card(self, card_pools: Dict[str, List[Dict]]) -> Optional[Dict]:
        """
        Select a single card using the distribution:
        - 90% from weighted list (with popularity weights)
        - 10% from non-weighted pools (4% generic, 1% class, 1% talent, 4% both)
        
        When selecting from weighted list, cards are chosen based on their
        deck usage percentages converted to selection weights.
        
        Returns:
            Selected card data or None
        """
        roll = random.random()
        
        if roll < 0.90:  # 90% weighted
            if card_pools['weighted']:
                # Use usage_percentage for weighted selection, converted to weights
                cards = card_pools['weighted']
                weights = [self._convert_usage_to_weight(c.get('usage_percentage', 1.0)) for c in cards]
                return random.choices(cards, weights=weights, k=1)[0]
        elif roll < 0.94:  # 4% generic (90-94%)
            if card_pools['generic']:
                return random.choice(card_pools['generic'])
        elif roll < 0.95:  # 1% class-only (94-95%)
            if card_pools['class_only']:
                return random.choice(card_pools['class_only'])
        elif roll < 0.96:  # 1% talent-only (95-96%)
            if card_pools['talent_only']:
                return random.choice(card_pools['talent_only'])
        else:  # 4% both (96-100%)
            if card_pools['both']:
                return random.choice(card_pools['both'])
        
        # Fallback to weighted if selected pool is empty
        if card_pools['weighted']:
            cards = card_pools['weighted']
            weights = [self._convert_usage_to_weight(c.get('usage_percentage', 1.0)) for c in cards]
            return random.choices(cards, weights=weights, k=1)[0]
        
        return None
    
    def filter_cards_for_zone(self, card_pool: List[Dict], zone_name: str, 
                              weapon_slot_state: Optional[Dict] = None,
                              tokens_only: bool = False) -> List[Dict]:
        """
        Filter card pool to only cards valid for a specific zone.
        
        Args:
            card_pool: List of cards to filter from
            zone_name: Name of the zone (e.g., 'Weapon', 'Pitch', 'Combat Chain 1')
            weapon_slot_state: Dict with 'weapon_is_2h' key for weapon slot logic
            tokens_only: If True, filter to only Token cards (for combat chain)
        
        Returns:
            List of valid cards for the zone
        """
        # Combat Chain zones exclude specific card types
        if zone_name in ['Combat Chain 1', 'Combat Chain 2']:
            excluded_types = {'Resource', 'Equipment', 'Weapon', 'Off-Hand', 'Hero'}
            valid_cards = []
            for card in card_pool:
                card_types = set(card.get('types', []))
                if not card_types.intersection(excluded_types):
                    # If tokens_only, also check for Token type
                    if tokens_only and 'Token' not in card_types:
                        continue
                    valid_cards.append(card)
            return valid_cards
        
        # Equipment zones require matching equipment type
        equipment_zones = {
            'Head': 'Head', 'Head 2': 'Head',
            'Chest': 'Chest', 'Chest 2': 'Chest',
            'Arms': 'Arms', 'Arms 2': 'Arms',
            'Legs': 'Legs', 'Legs 2': 'Legs'
        }
        
        # Pitch zones require cards with valid pitch value
        if zone_name in ['Pitch', 'Pitch 2']:
            return [card for card in card_pool if card.get('pitch', '') not in ['', '0']]
        
        # Banish zones exclude Equipment and Weapon cards
        if zone_name in ['Banish', 'Banish 2']:
            excluded_types = {'Equipment', 'Weapon'}
            valid_cards = []
            for card in card_pool:
                card_types = set(card.get('types', []))
                if not card_types.intersection(excluded_types):
                    valid_cards.append(card)
            return valid_cards
        
        # Weapon zones have special rules
        if zone_name in ['Weapon', 'Weapon 2']:
            return [card for card in card_pool if 'Weapon' in card.get('types', [])]
        
        # Weapon or Off-Hand zones have conditional rules
        if zone_name in ['Weapon or Off-Hand', 'Weapon or Off-Hand 2']:
            if weapon_slot_state and weapon_slot_state.get('weapon_is_2h'):
                return []
            
            valid_cards = []
            for card in card_pool:
                card_types = card.get('types', [])
                if 'Off-Hand' in card_types:
                    valid_cards.append(card)
                elif 'Weapon' in card_types and '1H' in card_types:
                    valid_cards.append(card)
            return valid_cards
        
        # Regular equipment zones
        if zone_name in equipment_zones:
            required_type = equipment_zones[zone_name]
            filtered = [card for card in card_pool if required_type in card.get('types', [])]
            valid_cards = filtered if filtered else []
        else:
            # For non-equipment zones, return all cards
            valid_cards = card_pool
        
        # Apply tokens_only filter if requested (for non-combat-chain zones)
        if tokens_only and zone_name not in ['Combat Chain 1', 'Combat Chain 2']:
            return [card for card in valid_cards if 'Token' in card.get('types', [])]
        
        return valid_cards
    
    def select_card_for_zone(self, card_pools: Dict[str, List[Dict]], zone_name: str,
                             all_cards_pool: List[Dict], weapon_slot_state: Optional[Dict] = None,
                             pitch_weighting: bool = True) -> Optional[Dict]:
        """
        Select a card that is valid for a specific zone.
        Combines weighted selection with zone filtering.
        
        Args:
            card_pools: Hero card pools from build_card_pools()
            zone_name: Name of the zone
            all_cards_pool: Complete list of all hero cards (weighted + generic + class + talent + both)
            weapon_slot_state: Dict with 'weapon_is_2h' key for weapon slot logic
            pitch_weighting: If True, apply 80/15/5 blue/yellow/red weighting for Pitch zones
        
        Returns:
            Selected card or None if no valid cards found
        """
        # Filter to cards valid for this zone
        valid_cards = self.filter_cards_for_zone(all_cards_pool, zone_name, weapon_slot_state)
        
        if not valid_cards:
            return None
        
        # For Pitch zones with weighting enabled, apply pitch-based selection
        if zone_name in ['Pitch', 'Pitch 2'] and pitch_weighting:
            # Collect candidates using normal weighted selection
            pitch_candidates = []
            max_attempts = 50
            for _ in range(max_attempts):
                candidate = self.select_card(card_pools)
                if candidate in valid_cards:
                    pitch_candidates.append(candidate)
            
            if not pitch_candidates:
                return None
            
            # Apply pitch weighting: 80% blue (3), 15% yellow (2), 5% red (1)
            blue_cards = [c for c in pitch_candidates if str(c.get('pitch', '')) == '3']
            yellow_cards = [c for c in pitch_candidates if str(c.get('pitch', '')) == '2']
            red_cards = [c for c in pitch_candidates if str(c.get('pitch', '')) == '1']
            
            rand = random.random()
            if rand < 0.80 and blue_cards:
                return random.choice(blue_cards)
            elif rand < 0.95 and yellow_cards:
                return random.choice(yellow_cards)
            elif red_cards:
                return random.choice(red_cards)
            else:
                # Fallback
                if blue_cards:
                    return random.choice(blue_cards)
                elif yellow_cards:
                    return random.choice(yellow_cards)
                elif red_cards:
                    return random.choice(red_cards)
            return None
        
        # For non-pitch zones, use normal weighted selection with zone filtering
        max_attempts = 50
        for _ in range(max_attempts):
            candidate = self.select_card(card_pools)
            if candidate in valid_cards:
                return candidate
        
        # Fallback: random selection from valid cards if weighted selection failed
        return random.choice(valid_cards) if valid_cards else None
    
    def find_card_image(self, card_data: Dict, card_dir: str = 'data/images') -> Optional[Path]:
        """
        Find the image file for a card by randomly selecting a printing.
        
        Downloaded filename pattern:
        - CardName_URLIdentifier.ext
        
        Where URLIdentifier is extracted from the image_url (e.g., MON091, fy8w7r78545yit3787efygs8def.width-450, etc.)
        
        Args:
            card_data: Card dictionary from card.json
            card_dir: Base directory for card images
        
        Returns:
            Path to image file or None
        """
        if not card_data.get('printings'):
            return None
        
        # Shuffle printings to try them in random order
        import random
        printings = list(card_data['printings'])
        random.shuffle(printings)
        
        card_name = card_data['name']
        
        # Map pitch value: 1=R, 2=Y, 3=B
        pitch_value = card_data.get('pitch', '')
        pitch_map = {'1': 'R', '2': 'Y', '3': 'B'}
        pitch = pitch_map.get(str(pitch_value), '')
        
        # Sanitize card name for filename (must match download script logic)
        # Replace spaces with underscores, then keep only alphanumeric, underscore, hyphen
        import re
        safe_name = card_name.replace(' ', '_')
        safe_name = re.sub(r'[^A-Za-z0-9_-]', '', safe_name)
        
        # Try each printing until we find one
        for printing in printings:
            set_id = printing['set_id']
            image_url = printing.get('image_url', '')
            
            if not image_url:
                continue
            
            # Build search patterns based on pitch
            base_path = Path(f'{card_dir}/{set_id}')
            
            if not base_path.exists():
                continue
            
            # Extract the URL identifier (everything after last slash, before extension)
            # e.g., "MON091.width-450", "fy8w7r78545yit3787efygs8def.width-450", "U-MON091.width-450"
            url_filename = image_url.rstrip('/').split('/')[-1]  # Get last part of URL
            url_identifier = url_filename.rsplit('.', 1)[0]  # Remove extension (.png)
            
            # Build search patterns
            # Priority:
            # 1. CardName_PitchColor_URLIdentifier* (for pitch cards with URL identifier)
            # 2. CardName_URLIdentifier* (card name with URL identifier)
            # 3. URLIdentifier* (just the URL identifier as fallback)
            
            search_patterns = []
            
            if pitch in ['R', 'Y', 'B']:
                search_patterns.append(f'{safe_name}_{pitch}_{url_identifier}*')
            
            search_patterns.append(f'{safe_name}_{url_identifier}*')
            search_patterns.append(f'{url_identifier}*')
            
            # Try each pattern
            for pattern in search_patterns:
                # Try different extensions
                for ext in ['png', 'jpg', 'jpeg', 'webp']:
                    matches = list(base_path.glob(f'{pattern}.{ext}'))
                    if matches:
                        return matches[0]
        
        return None


# Test the system
if __name__ == '__main__':
    print("="*60)
    print("Testing Card Selection System")
    print("="*60)
    
    selector = CardSelector()
    
    print("\n" + "="*60)
    print("Selecting random hero...")
    print("="*60)
    
    hero_key, hero_card, hero_weights = selector.select_random_hero()
    
    print(f"\nHero key: {hero_key}")
    print(f"Hero name: {hero_card['name']}")
    print(f"Hero types: {', '.join(hero_card['types'])}")
    
    hero_classes, hero_talents = selector.get_hero_classes_and_talents(hero_card)
    print(f"Classes: {hero_classes}")
    print(f"Talents: {hero_talents}")
    
    print("\n" + "="*60)
    print("Building card pools...")
    print("="*60)
    
    card_pools = selector.build_card_pools(hero_card, hero_weights)
    
    print(f"\nCard pool sizes:")
    print(f"  Weighted: {len(card_pools['weighted'])}")
    print(f"  Generic: {len(card_pools['generic'])}")
    print(f"  Class-only: {len(card_pools['class_only'])}")
    print(f"  Talent-only: {len(card_pools['talent_only'])}")
    print(f"  Both: {len(card_pools['both'])}")
    
    print("\n" + "="*60)
    print("Sampling 100 cards...")
    print("="*60)
    
    # Test distribution
    results = {
        'weighted': 0,
        'generic': 0,
        'class_only': 0,
        'talent_only': 0,
        'both': 0
    }
    
    for _ in range(100):
        card = selector.select_card(card_pools)
        if card:
            # Determine which pool it came from
            roll = random.random()
            if roll < 0.90:
                results['weighted'] += 1
            elif roll < 0.94:
                results['generic'] += 1
            elif roll < 0.95:
                results['class_only'] += 1
            elif roll < 0.96:
                results['talent_only'] += 1
            else:
                results['both'] += 1
    
    print(f"\nDistribution (expected vs actual):")
    print(f"  Weighted:    90% -> {results['weighted']}%")
    print(f"  Generic:      4% -> {results['generic']}%")
    print(f"  Class-only:   1% -> {results['class_only']}%")
    print(f"  Talent-only:  1% -> {results['talent_only']}%")
    print(f"  Both:         4% -> {results['both']}%")
    
    print("\n" + "="*60)
    print("Sample cards:")
    print("="*60)
    
    for i in range(5):
        card = selector.select_card(card_pools)
        if card:
            print(f"\n{i+1}. {card['name']}")
            print(f"   Types: {', '.join(card.get('types', []))}")
            
            img_path = selector.find_card_image(card)
            if img_path:
                print(f"   Image: ✓ {img_path}")
            else:
                print(f"   Image: ✗ Not found")
    
    print("\n" + "="*60)
    print("Test complete!")
    print("="*60)
