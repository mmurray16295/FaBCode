"""
Extend class definitions with new cards while preserving existing class order.
Creates a new core version (e.g., v2) by appending new cards to the end.

Usage:
    python scripts/class_management/extend_class_definitions.py
    python scripts/class_management/extend_class_definitions.py --dry-run
"""

import sys
import json
import hashlib
import argparse
from pathlib import Path
from datetime import datetime
from collections import OrderedDict


def compute_class_list_hash(names_list):
    """Compute SHA256 hash of class names list for verification."""
    names_str = str(names_list)
    return hashlib.sha256(names_str.encode()).hexdigest()[:16]


def load_existing_classes(yaml_path):
    """Load class names from existing core YAML."""
    import yaml
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    return data['names']


def load_card_names_from_json(card_json_path):
    """Extract unique card names from card.json."""
    with open(card_json_path, 'r', encoding='utf-8') as f:
        all_cards = json.load(f)
    
    # Extract unique names preserving first occurrence order
    unique_names = OrderedDict()
    for card in all_cards:
        name = card.get('name')
        if name and name not in unique_names:
            unique_names[name] = True
    
    return list(unique_names.keys())


def find_latest_core_version(class_defs_dir):
    """Find the highest version number of core class files."""
    versions = []
    for path in class_defs_dir.glob('core_classes_v*.yaml'):
        try:
            version = int(path.stem.split('_v')[-1])
            versions.append((version, path))
        except ValueError:
            continue
    
    if not versions:
        return None, None
    
    versions.sort(reverse=True)
    return versions[0]  # (version_num, path)


def extend_class_definitions(dry_run: bool = False):
    """
    Extend class definitions with new cards from card.json.
    Preserves existing class order and appends new cards to the end.
    """
    # Paths
    root = Path(__file__).parent.parent.parent
    class_defs_dir = Path(__file__).parent
    card_json_path = root / 'data' / 'card.json'
    
    print("=" * 80)
    print("EXTEND CLASS DEFINITIONS")
    print("=" * 80)
    
    # Find latest core version
    latest_version, latest_path = find_latest_core_version(class_defs_dir)
    
    if latest_version is None:
        print("\n❌ ERROR: No core class definitions found!")
        print(f"   Expected files like: core_classes_v1.yaml")
        print(f"   In directory: {class_defs_dir}")
        sys.exit(1)
    
    print(f"\nLatest core version: v{latest_version}")
    print(f"Path: {latest_path}")
    
    # Load existing classes
    print(f"\nLoading existing classes from v{latest_version}...")
    try:
        existing_classes = load_existing_classes(latest_path)
        print(f"✓ Loaded {len(existing_classes)} existing classes")
        print(f"  First: {existing_classes[0]}")
        print(f"  Last: {existing_classes[-1]}")
    except Exception as e:
        print(f"❌ ERROR: Failed to load existing classes: {e}")
        sys.exit(1)
    
    # Load card names from card.json
    print(f"\nLoading card names from card.json...")
    if not card_json_path.exists():
        print(f"❌ ERROR: card.json not found at {card_json_path}")
        sys.exit(1)
    
    try:
        all_card_names = load_card_names_from_json(card_json_path)
        print(f"✓ Found {len(all_card_names)} unique card names in card.json")
    except Exception as e:
        print(f"❌ ERROR: Failed to load card.json: {e}")
        sys.exit(1)
    
    # Find new cards (in card.json but not in existing classes)
    existing_set = set(existing_classes)
    new_cards = [name for name in all_card_names if name not in existing_set]
    
    print(f"\n" + "=" * 80)
    print(f"ANALYSIS")
    print(f"=" * 80)
    print(f"Existing classes: {len(existing_classes)}")
    print(f"Card.json cards: {len(all_card_names)}")
    print(f"New cards to add: {len(new_cards)}")
    
    if not new_cards:
        print(f"\n✓ No new cards found!")
        print(f"   All cards in card.json are already in v{latest_version}")
        print(f"   No need to create v{latest_version + 1}")
        return
    
    # Sort new cards alphabetically (to maintain consistency within new additions)
    new_cards_sorted = sorted(new_cards)
    
    print(f"\nNew cards to append (alphabetically sorted):")
    for i, card in enumerate(new_cards_sorted[:10], start=len(existing_classes)):
        print(f"  Class {i}: {card}")
    if len(new_cards_sorted) > 10:
        print(f"  ... ({len(new_cards_sorted) - 10} more)")
        for i, card in enumerate(new_cards_sorted[-3:], start=len(existing_classes) + len(new_cards_sorted) - 3):
            print(f"  Class {i}: {card}")
    
    # Build extended class list
    extended_classes = existing_classes + new_cards_sorted
    
    # Compute hashes
    old_hash = compute_class_list_hash(existing_classes)
    new_hash = compute_class_list_hash(extended_classes)
    
    print(f"\nClass order hashes:")
    print(f"  v{latest_version} hash: {old_hash}")
    print(f"  v{latest_version + 1} hash: {new_hash}")
    
    # Create new version
    new_version = latest_version + 1
    new_path = class_defs_dir / f'core_classes_v{new_version}.yaml'
    
    if new_path.exists():
        print(f"\n⚠️  WARNING: {new_path.name} already exists!")
        if not dry_run:
            response = input("    Overwrite? (y/n): ").strip().lower()
            if response != 'y':
                print("    Cancelled.")
                return
    
    print(f"\n" + "=" * 80)
    if dry_run:
        print(f"DRY RUN - Would create:")
    else:
        print(f"CREATING NEW CORE VERSION")
    print(f"=" * 80)
    print(f"Version: v{new_version}")
    print(f"Path: {new_path}")
    print(f"Classes: {len(extended_classes)} ({len(existing_classes)} existing + {len(new_cards_sorted)} new)")
    
    if dry_run:
        print(f"\n✓ Dry run complete. Use without --dry-run to create v{new_version}")
        return
    
    # Generate YAML content with metadata header
    yaml_content = f"""# YOLO Dataset Configuration - Card Recognition
# Core Class Definitions v{new_version}
# ⚠️  WARNING: DO NOT MODIFY THIS FILE DIRECTLY
# This is a protected, version-locked class order file
#
# Version: v{new_version}
# Date: {datetime.now().strftime('%Y-%m-%d')}
# Classes: {len(extended_classes)}
# Based on: v{latest_version} ({len(existing_classes)} classes)
# Added: {len(new_cards_sorted)} new classes
# Hash: {new_hash}
#
# Model compatibility:
# - v{latest_version}: (existing trained models)
# - v{new_version}: (future models trained with new classes)
#
# Class ranges:
# - [0-{len(existing_classes)-1}]: Original classes from v{latest_version}
# - [{len(existing_classes)}-{len(extended_classes)-1}]: New classes added in v{new_version}

train: train/images
val: valid/images
test: test/images

nc: {len(extended_classes)}
names: {extended_classes}
"""
    
    # Write new core version
    print(f"\n📝 Writing {new_path.name}...")
    new_path.write_text(yaml_content, encoding='utf-8')
    print(f"✓ Created")
    
    # Make read-only
    print(f"\n🔒 Setting read-only protection...")
    try:
        import os
        # Windows
        os.system(f'attrib +r "{new_path}"')
        print(f"✓ File is now read-only")
    except Exception as e:
        print(f"⚠️  Could not set read-only: {e}")
        print(f"   Please manually protect this file!")
    
    # Summary
    print(f"\n" + "=" * 80)
    print(f"SUCCESS")
    print(f"=" * 80)
    print(f"Created: core_classes_v{new_version}.yaml")
    print(f"Location: {new_path}")
    print(f"Status: Read-only, ready for git commit")
    print(f"\nNext steps:")
    print(f"1. Commit to git:")
    print(f"   git add {new_path}")
    print(f"   git commit -m 'Add core_classes_v{new_version}.yaml ({len(new_cards_sorted)} new cards)'")
    print(f"\n2. Sync working copy:")
    print(f"   python scripts/class_management/sync_data_yaml.py --version {new_version}")
    print(f"\n3. Continue training with new classes:")
    print(f"   python scripts/train_yolo11x.py --resume <existing_model.pt>")


def main():
    parser = argparse.ArgumentParser(
        description="Extend class definitions with new cards from card.json",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
This script:
1. Finds the latest core_classes_v*.yaml
2. Loads existing class order
3. Finds new cards in card.json
4. Creates new version with preserved order + new cards appended
5. Makes new version read-only
6. Instructs you to commit to git

Example:
  # Preview changes
  python scripts/class_management/extend_class_definitions.py --dry-run
  
  # Create new version
  python scripts/class_management/extend_class_definitions.py
        """
    )
    
    parser.add_argument(
        '--dry-run',
        action='store_true',
        help='Preview changes without creating new version'
    )
    
    args = parser.parse_args()
    
    extend_class_definitions(args.dry_run)


if __name__ == '__main__':
    main()
