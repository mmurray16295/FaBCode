"""
Hero-Aware Confidence Booster
Adjusts detection confidence based on card meta-usage within detected hero's deck.
"""

import json
from pathlib import Path
from typing import Dict, Optional, Tuple


class ConfidenceBooster:
    """
    Adjusts card detection confidence based on hero-specific meta usage data.
    
    The core insight: Generic cards appear more in training data (used by all heroes),
    while class-specific cards are underrepresented. This booster compensates by 
    increasing confidence for cards likely in the detected hero's deck.
    """
    
    def __init__(self, weights_file: str, enabled: bool = False):
        """
        Initialize the confidence booster.
        
        Args:
            weights_file: Path to card_weights_all_printings.json
            enabled: Whether boosting is active (toggleable mode)
        """
        self.enabled = enabled
        self.weights_data = self._load_weights(weights_file)
        self.current_hero = None
        self.current_format = "cc"  # Default to Classic Constructed
        self.card_usage_cache = {}
        
    def _load_weights(self, weights_file: str) -> dict:
        """Load the card weights JSON."""
        try:
            with open(weights_file, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            print(f"[ConfidenceBooster] Warning: Could not load weights file: {e}")
            return {"formats": {"cc": {}, "blitz": {}}}
    
    def set_active_hero(self, hero_name: str, format_name: str = "cc"):
        """
        Set the currently detected hero to load their deck profile.
        
        Args:
            hero_name: Name of the hero (e.g., "Dorinthea Ironsong")
            format_name: Format ("cc" or "blitz")
        """
        self.current_hero = hero_name
        self.current_format = format_name
        self.card_usage_cache = self._build_usage_cache(hero_name, format_name)
        
        if self.enabled and self.card_usage_cache:
            print(f"[ConfidenceBooster] Loaded {len(self.card_usage_cache)} cards for {hero_name} ({format_name})")
    
    def _build_usage_cache(self, hero_name: str, format_name: str) -> Dict[str, float]:
        """
        Build a quick-lookup cache of card name -> usage percentage for the hero.
        
        Returns:
            Dict mapping card_name to usage_percentage
        """
        cache = {}
        
        try:
            hero_data = self.weights_data['formats'][format_name].get(hero_name)
            if not hero_data:
                return cache
            
            # Aggregate all cards across all sections
            for section_name, cards in hero_data['sections'].items():
                for card in cards:
                    card_name = card['card_name']
                    usage_pct = card['usage_percentage']
                    
                    # Keep highest usage if card appears in multiple sections
                    if card_name not in cache or usage_pct > cache[card_name]:
                        cache[card_name] = usage_pct
                        
        except Exception as e:
            print(f"[ConfidenceBooster] Warning: Error building cache: {e}")
        
        return cache
    
    def get_confidence_multiplier(self, card_name: str, raw_confidence: float) -> float:
        """
        Calculate the confidence multiplier for a detected card.
        
        Strategy:
        - Cards in hero's deck with high usage (50%+) get significant boost
        - Cards in hero's deck with medium usage (20-50%) get moderate boost  
        - Cards in hero's deck with low usage (5-20%) get small boost
        - Cards NOT in hero's deck get slight penalty (might be generic/wrong hero)
        - Only apply to borderline detections (0.20-0.45 confidence range)
        
        Args:
            card_name: Name of the detected card
            raw_confidence: Raw confidence from YOLO (0.0-1.0)
            
        Returns:
            Multiplier to apply to confidence (e.g., 1.15 = +15%)
        """
        # If disabled or no hero selected, return 1.0 (no change)
        if not self.enabled or not self.current_hero or not self.card_usage_cache:
            return 1.0
        
        # Only adjust borderline detections (high confidence is already solid)
        if raw_confidence < 0.20 or raw_confidence > 0.45:
            return 1.0
        
        # Check if card is in this hero's deck
        usage_pct = self.card_usage_cache.get(card_name)
        
        if usage_pct is None:
            # Card NOT in this hero's typical deck
            # Likely a generic card or wrong hero detection
            return 0.92  # -8% penalty
        
        # Card IS in hero's deck - boost based on meta usage
        if usage_pct >= 70:
            return 1.20  # Staple card: +20% boost
        elif usage_pct >= 50:
            return 1.15  # Very common: +15% boost
        elif usage_pct >= 30:
            return 1.10  # Common: +10% boost
        elif usage_pct >= 15:
            return 1.05  # Playable: +5% boost
        else:
            return 1.02  # Tech card: +2% boost (it's real, just rare)
    
    def adjust_confidence(self, card_name: str, raw_confidence: float) -> Tuple[float, str]:
        """
        Adjust confidence for a detection and provide reasoning.
        
        Args:
            card_name: Name of the detected card
            raw_confidence: Raw confidence from YOLO
            
        Returns:
            Tuple of (adjusted_confidence, reason_string)
        """
        multiplier = self.get_confidence_multiplier(card_name, raw_confidence)
        adjusted = raw_confidence * multiplier
        
        # Clamp to valid range
        adjusted = max(0.0, min(1.0, adjusted))
        
        # Generate reason string
        if not self.enabled:
            reason = "boosting disabled"
        elif multiplier == 1.0:
            if raw_confidence < 0.20:
                reason = "conf too low"
            elif raw_confidence > 0.45:
                reason = "conf already high"
            else:
                reason = "no hero context"
        elif multiplier > 1.0:
            usage_pct = self.card_usage_cache.get(card_name, 0)
            reason = f"in {self.current_hero} deck ({usage_pct:.0f}%)"
        else:
            reason = f"not typical for {self.current_hero}"
        
        return adjusted, reason
    
    def toggle(self, enabled: Optional[bool] = None) -> bool:
        """
        Toggle the booster on/off or set explicitly.
        
        Args:
            enabled: If None, toggle current state. If bool, set to that state.
            
        Returns:
            New enabled state
        """
        if enabled is None:
            self.enabled = not self.enabled
        else:
            self.enabled = enabled
        
        return self.enabled
    
    def get_stats(self) -> dict:
        """Get current booster statistics."""
        return {
            "enabled": self.enabled,
            "current_hero": self.current_hero,
            "current_format": self.current_format,
            "cards_loaded": len(self.card_usage_cache),
            "has_context": self.current_hero is not None and len(self.card_usage_cache) > 0
        }


# Example usage and testing
if __name__ == "__main__":
    # Initialize booster
    weights_file = Path(__file__).parent.parent.parent / 'data' / 'card_weights_all_printings.json'
    booster = ConfidenceBooster(str(weights_file), enabled=True)
    
    # Set active hero
    booster.set_active_hero("Dorinthea Ironsong", "cc")
    
    # Test some detections
    test_cases = [
        ("Fyendal's Spring Tunic", 0.28),  # Should boost (high usage equipment)
        ("Courage of Bladehold", 0.32),    # Should boost (warrior card)
        ("Sink Below", 0.29),              # Might penalize if not in Dori's deck
        ("Command and Conquer", 0.26),     # Should boost (common in Dori)
        ("Unknown Card", 0.25),            # Should penalize (not in deck)
        ("High Conf Card", 0.85),          # Should not change (already confident)
        ("Low Conf Card", 0.12),           # Should not change (too low to help)
    ]
    
    print("\n" + "="*70)
    print("CONFIDENCE BOOSTER TEST")
    print("="*70)
    print(f"Hero: {booster.current_hero}")
    print(f"Cards loaded: {len(booster.card_usage_cache)}")
    print()
    
    for card_name, raw_conf in test_cases:
        adj_conf, reason = booster.adjust_confidence(card_name, raw_conf)
        change = ((adj_conf - raw_conf) / raw_conf * 100) if raw_conf > 0 else 0
        
        status = "✓" if adj_conf > raw_conf else "✗" if adj_conf < raw_conf else "→"
        
        print(f"{status} {card_name:30s}")
        print(f"   Raw: {raw_conf:.3f} → Adjusted: {adj_conf:.3f} ({change:+.1f}%)")
        print(f"   Reason: {reason}")
        print()
