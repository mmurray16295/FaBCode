"""
Smooth card selection system for even distribution across all cards.

This selector tracks usage counts and randomly selects from the LEAST-USED legal cards,
creating a "draw without replacement" effect over time.

Key differences from card_selector.py:
- Respects hero legality (class/talent matching) like weighted selector
- Tracks usage counts globally - selects randomly from least-used legal options
- "Draw without replacement" - cycles through all legal cards before repeating
- Global usage tracking persisted to JSON
- Designed for "filling in gaps" after weighted generation
- Saves to synthetic_smooth/ instead of synthetic/
- NO weighted selection - always picks from card.json directly (100% unweighted)
"""

import json
import random
from typing import Dict, List, Set, Optional, Tuple
from pathlib import Path
import re

# Classes and talents (same as card_selector.py)
ALL_CLASSES = {'Warrior', 'Brute', 'Guardian', 'Ninja', 'Ranger', 'Assassin', 
               'Wizard', 'Runeblade', 'Mechanologist', 'Merchant', 'Illusionist', 
               'Shapeshifter', 'Bard', 'Mystic', 'Pirate', 'Necromancer', 'Thief', 
               'Adjudicator'}
ALL_TALENTS = {'Draconic', 'Elemental', 'Light', 'Shadow', 'Earth', 'Ice', 
               'Lightning', 'Royal', 'Chi', 'Chaos', 'Arcane', 'Revered', 'Reviled'}


def select_format() -> str:
    """Select a format with weighted probability: CC 70%, Blitz 30%."""
    rand = random.random()
    if rand < 0.70:
        return 'cc'
    else:
        return 'blitz'


def is_card_legal_for_format(card: Dict, format: str) -> bool:
    """Check if a card is legal in the specified format.
    
    Args:
        card: Card dict with legality fields
        format: One of 'cc' or 'blitz'
    
    Returns:
        True if card is legal in the format
    """
    format_key = f'{format}_legal'
    return card.get(format_key) == True


class SmoothCardSelector:
    """Card selector that ensures even distribution across all cards."""
    
    def __init__(self, card_json_path: str = 'data/card.json',
                 weights_path: str = 'data/card_weights_all_printings.json',
                 state_file: str = 'smooth_selector_state.json',
                 enable_state_persistence: bool = True):
        """Initialize with data files and load usage state.
        
        Args:
            card_json_path: Path to card.json
            weights_path: Path to weights file (for hero list)
            state_file: Path to state persistence file
            enable_state_persistence: If False, disables disk writes (for testing/performance)
        """
        print("Loading card database...")
        with open(card_json_path, 'r', encoding='utf-8') as f:
            self.all_cards = json.load(f)
        
        # We don't use weights, but load it to get hero list
        print("Loading hero list from weights file...")
        with open(weights_path, 'r', encoding='utf-8') as f:
            self.weights_data = json.load(f)
        
        # Create card lookup by name for fast access
        self.card_lookup = {}
        for card in self.all_cards:
            name = card['name']
            if name not in self.card_lookup:
                self.card_lookup[name] = card
        
        # Extract hero names from weights (just to know which heroes exist)
        self.heroes_by_format = {}
        all_hero_names = set()
        for format_code, heroes in self.weights_data.get('formats', {}).items():
            self.heroes_by_format[format_code] = list(heroes.keys())
            all_hero_names.update(heroes.keys())
        
        print(f"Loaded {len(self.all_cards)} cards, {len(self.card_lookup)} unique names")
        print(f"Found {len(all_hero_names)} total heroes across all formats")
        
        # Load or initialize usage tracking
        self.state_file = state_file
        self.enable_state_persistence = enable_state_persistence
        self.hero_counts = {}  # {format: {hero_name: count}}
        self.card_counts = {}  # {card_name: count}
        self.load_state()
    
    def load_state(self):
        """Load usage counts from JSON file."""
        if Path(self.state_file).exists():
            try:
                with open(self.state_file, 'r', encoding='utf-8') as f:
                    data = json.load(f)
                    self.hero_counts = data.get('hero_counts', {})
                    self.card_counts = data.get('card_counts', {})
                print(f"Loaded smooth selector state: {len(self.card_counts)} cards tracked")
            except Exception as e:
                print(f"Warning: Could not load state file: {e}")
                print("Starting with fresh counts")
        else:
            print("No state file found - starting with fresh counts")
    
    def save_state(self):
        """Save usage counts to JSON file (only if persistence enabled)."""
        if not self.enable_state_persistence:
            return
        
        try:
            data = {
                'hero_counts': self.hero_counts,
                'card_counts': self.card_counts
            }
            with open(self.state_file, 'w', encoding='utf-8') as f:
                json.dump(data, f, indent=2)
        except Exception as e:
            print(f"Warning: Could not save state file: {e}")
    
    def reset_state(self):
        """Reset all usage counts to zero."""
        self.hero_counts = {}
        self.card_counts = {}
        self.save_state()
        print("Smooth selector state reset to zero")
    
    def select_random_hero(self, format: str = None) -> Tuple[str, Dict, Dict]:
        """
        Select the least-used hero for the given format.
        Always returns unweighted (empty weights dict).
        
        Args:
            format: Optional format to filter by ('cc' or 'blitz')
        
        Returns:
            (hero_name, hero_card_data, {})
        """
        # Get hero names for this format
        hero_names = self.heroes_by_format.get(format, []) if format else []
        
        if not hero_names:
            raise ValueError(f"No heroes found for format: {format}")
        
        # Initialize format counts if needed
        if format not in self.hero_counts:
            self.hero_counts[format] = {}
        
        # Find least-used heroes
        min_count = float('inf')
        least_used = []
        
        for hero_name in hero_names:
            count = self.hero_counts[format].get(hero_name, 0)
            if count < min_count:
                min_count = count
                least_used = [hero_name]
            elif count == min_count:
                least_used.append(hero_name)
        
        # Random tie-breaking
        hero_name = random.choice(least_used)
        
        # Increment count
        self.hero_counts[format][hero_name] = self.hero_counts[format].get(hero_name, 0) + 1
        self.save_state()
        
        # Get hero card data from card.json
        hero_card = self.card_lookup.get(hero_name)
        if not hero_card:
            raise ValueError(f"Hero card not found in card.json: {hero_name}")
        
        # Always return empty weights dict (unweighted)
        return (hero_name, hero_card, {})
    
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
                
                # Add valid talents
                for talent in essence_talents:
                    if talent in ALL_TALENTS:
                        talents.add(talent)
        
        return classes, talents
    
    def build_card_pools(self, hero_card: Dict, hero_weights: Dict, format: str) -> Dict[str, List[Dict]]:
        """
        Build categorized card pools for a given hero.
        For smooth selector, we ignore hero_weights and build pools from card.json only.
        
        Args:
            hero_card: Hero card data from card.json
            hero_weights: Ignored (always empty dict for smooth selector)
            format: Game format ('cc', 'blitz')
        
        Returns:
            Dict with 'weighted' (always empty), 'generic', 'class_only', 'talent_only', 'both' lists
        """
        hero_classes, hero_talents = self.get_hero_classes_and_talents(hero_card)
        
        # Build non-weighted pools only (no weighted pool for smooth selector)
        generic_cards = []
        class_only_cards = []
        talent_only_cards = []
        both_cards = []
        
        for card in self.all_cards:
            # Skip heroes and non-legal cards
            if 'Hero' in card.get('types', []) or not is_card_legal_for_format(card, format):
                continue
            
            card_types = set(card.get('types', []))
            
            # Generic cards only (tokens will be checked for class/talent matching below)
            if 'Generic' in card_types:
                generic_cards.append(card)
                continue
            
            # Companion cards - match if card type matches hero name
            if 'Companion' in card_types:
                hero_name_first_word = hero_card['name'].split(',')[0].split()[0].lower()
                companion_types_lower = {t.lower() for t in card_types}
                if hero_name_first_word in companion_types_lower:
                    class_only_cards.append(card)
                    continue
                continue
            
            # Check class/talent matching
            card_classes = card_types & ALL_CLASSES
            card_talents = card_types & ALL_TALENTS
            
            # If hero has no talents, exclude cards with ANY talent
            if not hero_talents and card_talents:
                continue
            
            # Check if card matches hero
            card_classes_match = not card_classes or bool(card_classes & hero_classes)
            card_talents_match = not card_talents or bool(card_talents & hero_talents)
            
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
            'weighted': [],  # Always empty for smooth selector
            'generic': generic_cards,
            'class_only': class_only_cards,
            'talent_only': talent_only_cards,
            'both': both_cards
        }
    
    def get_hero_cards_pool(self, hero_card: Dict, format: str) -> List[Dict]:
        """
        Get all legal cards for a hero (from card.json only).
        
        Args:
            hero_card: Hero card dictionary
            format: Format code ('cc' or 'blitz')
        
        Returns:
            List of legal card dictionaries
        """
        hero_class = hero_card.get('class')
        hero_talents = hero_card.get('talents', [])
        if isinstance(hero_talents, str):
            hero_talents = [hero_talents]
        
        legal_cards = []
        for card in self.all_cards:
            # Skip heroes
            if 'Hero' in card.get('types', []):
                continue
            
            # Check format legality
            if not is_card_legal_for_format(card, format):
                continue
            
            # Check class/talent legality
            card_class = card.get('class')
            card_talents = card.get('talents', [])
            if isinstance(card_talents, str):
                card_talents = [card_talents]
            
            # Generic cards are always legal
            if card_class == 'Generic':
                legal_cards.append(card)
                continue
            
            # Class match
            if card_class == hero_class:
                legal_cards.append(card)
                continue
            
            # Talent match
            if any(t in hero_talents for t in card_talents):
                legal_cards.append(card)
                continue
        
        return legal_cards
    
    def select_least_used_card(self, candidates: List[Dict]) -> Optional[Dict]:
        """
        Select randomly from the least-used cards in candidates (optimized single-pass).
        "Draw without replacement" - cycles through all cards before repeating.
        
        Args:
            candidates: List of valid card dicts
        
        Returns:
            Selected card dict, or None if no candidates
        """
        if not candidates:
            return None
        
        # Single-pass algorithm: find min count and collect least-used cards simultaneously
        min_count = float('inf')
        least_used_tier = []
        
        for card in candidates:
            card_name = card['name']
            count = self.card_counts.get(card_name, 0)
            
            if count < min_count:
                # Found new minimum - reset tier
                min_count = count
                least_used_tier = [card]
            elif count == min_count:
                # Tied for minimum - add to tier
                least_used_tier.append(card)
        
        # Randomly select from the least-used tier
        selected_card = random.choice(least_used_tier)
        
        # Increment count
        card_name = selected_card['name']
        self.card_counts[card_name] = self.card_counts.get(card_name, 0) + 1
        self.save_state()
        
        return selected_card
    
    def select_card_for_zone(self, card_pools: Dict[str, List[Dict]], zone_name: str,
                            all_cards_pool: List[Dict], weapon_slot_state: Optional[Dict] = None,
                            pitch_weighting: bool = True) -> Optional[Dict]:
        """
        Select the least-used card for a zone.
        
        Args:
            card_pools: Hero card pools from build_card_pools() (not used directly)
            zone_name: Zone to select for
            all_cards_pool: Complete list of all hero cards (generic + class + talent + both)
            weapon_slot_state: Dict with 'weapon_is_2h' key for weapon slot logic
            pitch_weighting: If True, apply 80/15/5 blue/yellow/red weighting for Pitch zones
        
        Returns:
            Selected card dict, or None if no valid cards
        """
        # Filter to cards valid for this zone
        valid_cards = self.filter_cards_for_zone(all_cards_pool, zone_name, weapon_slot_state)
        
        if not valid_cards:
            return None
        
        # Apply pitch weighting if requested and this is a Pitch zone
        if pitch_weighting and 'Pitch' in zone_name:
            # Get pitch values for all valid cards
            pitch_groups = {'3': [], '2': [], '1': [], '0': []}  # Blue, Yellow, Red, No pitch
            for card in valid_cards:
                pitch = str(card.get('pitch', '0'))
                if pitch in pitch_groups:
                    pitch_groups[pitch].append(card)
                else:
                    pitch_groups['0'].append(card)
            
            # Apply 80/15/5 weighting (Blue 80%, Yellow 15%, Red 5%)
            roll = random.random()
            if roll < 0.80 and pitch_groups['3']:
                valid_cards = pitch_groups['3']
            elif roll < 0.95 and pitch_groups['2']:
                valid_cards = pitch_groups['2']
            elif pitch_groups['1']:
                valid_cards = pitch_groups['1']
            # If no cards in selected pitch, fall back to all valid cards
        
        return self.select_least_used_card(valid_cards)
    
    def select_card(self, card_pools: Dict[str, List[Dict]]) -> Optional[Dict]:
        """
        Select a single card using even distribution logic (no weighted preference).
        For smooth selector: combine all pools and select least-used.
        
        Args:
            card_pools: Dict with 'weighted' (empty), 'generic', 'class_only', 'talent_only', 'both'
        
        Returns:
            Selected card data or None
        """
        # Combine all non-empty pools (weighted is always empty for smooth selector)
        all_cards = []
        for pool_name in ['generic', 'class_only', 'talent_only', 'both']:
            all_cards.extend(card_pools.get(pool_name, []))
        
        if not all_cards:
            return None
        
        # Select from least-used tier
        return self.select_least_used_card(all_cards)
    
    def filter_cards_for_zone(self, cards: List[Dict], zone_name: str,
                             weapon_slot_state: Optional[Dict] = None,
                             tokens_only: bool = False) -> List[Dict]:
        """
        Filter cards valid for a specific zone (updated to match CardSelector signature).
        
        Args:
            cards: List of card dicts to filter
            zone_name: Name of the zone
            weapon_slot_state: Dict with 'weapon_is_2h' key for weapon slot logic
            tokens_only: If True, only return token cards
        
        Returns:
            List of valid cards for this zone
        """
        valid_cards = []
        weapon_is_2h = weapon_slot_state.get('weapon_is_2h', False) if weapon_slot_state else False
        
        for card in cards:
            types = card.get('types', [])
            
            # Token filter
            if tokens_only:
                if 'Token' not in types:
                    continue
            
            # Zone-specific filtering
            if zone_name == 'Hero':
                if 'Hero' in types:
                    valid_cards.append(card)
            
            elif zone_name in ['Weapon', 'Weapon 2']:
                if 'Weapon' in types:
                    valid_cards.append(card)
            
            elif zone_name in ['Weapon or Off-Hand', 'Weapon or Off-Hand 2']:
                if weapon_is_2h:
                    continue
                if '2H' in types:
                    continue
                if 'Weapon' in types or 'Off-Hand' in types:
                    valid_cards.append(card)
            
            elif zone_name in ['Head', 'Head 2', 'Chest', 'Chest 2', 'Arms', 'Arms 2', 'Legs', 'Legs 2']:
                # Map zone name to equipment type (strip player number)
                equipment_type = zone_name.replace(' 2', '')
                if 'Equipment' in types and equipment_type in types:
                    valid_cards.append(card)
            
            elif zone_name in ['Arsenal', 'Deck', 'Hand', 'Graveyard', 'Graveyard 2', 'Pitch', 'Pitch 2', 
                             'Banish', 'Banish 2', 'Combat Chain 1', 'Combat Chain 2']:
                # Most card types valid, exclude Equipment/Weapon/Hero
                if not any(t in types for t in ['Equipment', 'Weapon', 'Hero']):
                    valid_cards.append(card)
        
        return valid_cards
    
    def find_card_image(self, card_data: Dict, card_dir: str = 'data/images') -> Optional[Path]:
        """
        Find the image file for a card by randomly selecting a printing.
        (Same logic as CardSelector - this is image lookup, not selection)
        
        Args:
            card_data: Card dictionary from card.json
            card_dir: Base directory for card images
        
        Returns:
            Path to image file or None
        """
        if not card_data.get('printings'):
            return None
        
        # Shuffle printings to try them in random order
        printings = list(card_data['printings'])
        random.shuffle(printings)
        
        card_name = card_data['name']
        
        # Map pitch value
        pitch_value = card_data.get('pitch', '')
        pitch_map = {'1': 'R', '2': 'Y', '3': 'B'}
        pitch = pitch_map.get(str(pitch_value), '')
        
        # Sanitize card name for filename
        safe_name = card_name.replace(' ', '_')
        safe_name = re.sub(r'[^A-Za-z0-9_-]', '', safe_name)
        
        # Try each printing
        for printing in printings:
            set_id = printing['set_id']
            image_url = printing.get('image_url', '')
            
            if not image_url:
                continue
            
            base_path = Path(f'{card_dir}/{set_id}')
            if not base_path.exists():
                continue
            
            # Extract URL identifier
            url_filename = image_url.rstrip('/').split('/')[-1]
            url_identifier = url_filename.rsplit('.', 1)[0]
            
            # Build search patterns
            search_patterns = []
            if pitch in ['R', 'Y', 'B']:
                search_patterns.append(f'{safe_name}_{pitch}_{url_identifier}*')
            search_patterns.append(f'{safe_name}_{url_identifier}*')
            search_patterns.append(f'{url_identifier}*')
            
            # Try each pattern
            for pattern in search_patterns:
                for ext in ['.png', '.jpg', '.jpeg', '.webp']:
                    matches = list(base_path.glob(f'{pattern}{ext}'))
                    if matches:
                        return matches[0]
        
        return None

    
    def find_card_image(self, card_data: Dict, card_dir: str = 'data/images') -> Optional[Path]:
        """
        Find the image file for a card by randomly selecting a printing.
        (Same implementation as CardSelector for compatibility)
        
        Args:
            card_data: Card dictionary from card.json
            card_dir: Base directory for card images
        
        Returns:
            Path to image file or None
        """
        if not card_data.get('printings'):
            return None
        
        # Shuffle printings to try them in random order
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
            
            # Extract the URL identifier
            url_filename = image_url.rstrip('/').split('/')[-1]
            url_identifier = url_filename.rsplit('.', 1)[0]
            
            # Build search patterns
            search_patterns = []
            
            if pitch in ['R', 'Y', 'B']:
                search_patterns.append(f'{safe_name}_{pitch}_{url_identifier}*')
            
            search_patterns.append(f'{safe_name}_{url_identifier}*')
            search_patterns.append(f'{url_identifier}*')
            
            # Try each pattern
            for pattern in search_patterns:
                for ext in ['png', 'jpg', 'jpeg', 'webp']:
                    matches = list(base_path.glob(f'{pattern}.{ext}'))
                    if matches:
                        return matches[0]
        
        return None
