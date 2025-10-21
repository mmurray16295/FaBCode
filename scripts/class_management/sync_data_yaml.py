"""
Sync data.yaml working copy from protected core class definitions.
This script safely regenerates data/synthetic/data.yaml from the core version.

Usage:
    python scripts/class_management/sync_data_yaml.py --version 1
    python scripts/class_management/sync_data_yaml.py --version 2 --force
"""

import sys
import shutil
import hashlib
import argparse
from pathlib import Path


def compute_class_list_hash(yaml_path):
    """Compute SHA256 hash of just the class names list for verification."""
    import yaml
    
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Hash just the names list (order matters!)
    names_str = str(data['names'])
    return hashlib.sha256(names_str.encode()).hexdigest()[:16]


def load_yaml_metadata(yaml_path):
    """Extract version metadata from YAML file."""
    with open(yaml_path, 'r', encoding='utf-8') as f:
        lines = [line.strip() for line in f.readlines()[:20] if line.startswith('#')]
    
    metadata = {}
    for line in lines:
        if 'Version:' in line:
            metadata['version'] = line.split('Version:')[-1].strip()
        elif 'Classes:' in line:
            metadata['num_classes'] = line.split('Classes:')[-1].strip()
        elif 'Date:' in line:
            metadata['date'] = line.split('Date:')[-1].strip()
        elif 'Hash:' in line:
            metadata['hash'] = line.split('Hash:')[-1].strip()
    
    return metadata


def sync_data_yaml(version: int, force: bool = False):
    """
    Copy core class definition to working data.yaml location.
    
    Args:
        version: Core version to use (1, 2, etc.)
        force: Overwrite existing data.yaml without confirmation
    """
    # Paths
    root = Path(__file__).parent.parent.parent
    script_dir = Path(__file__).parent
    core_path = script_dir / f'core_classes_v{version}.yaml'
    working_path = root / 'data' / 'synthetic' / 'data.yaml'
    
    # Validate core version exists
    if not core_path.exists():
        print(f"❌ ERROR: Core version {version} not found!")
        print(f"   Expected: {core_path}")
        print(f"\n   Available versions:")
        for v_path in script_dir.glob('core_classes_v*.yaml'):
            print(f"   - {v_path.name}")
        sys.exit(1)
    
    # Check if core is read-only (good!)
    try:
        if core_path.stat().st_file_attributes & 0x1:  # FILE_ATTRIBUTE_READONLY
            print(f"✓ Core version is read-only (protected)")
    except:
        pass  # Not on Windows or other filesystem
    
    # Load metadata
    print(f"\n{'='*70}")
    print(f"SYNC DATA.YAML FROM CORE VERSION")
    print(f"{'='*70}")
    
    metadata = load_yaml_metadata(core_path)
    print(f"\nCore Version: {metadata.get('version', 'v' + str(version))}")
    print(f"Classes: {metadata.get('num_classes', 'unknown')}")
    print(f"Date: {metadata.get('date', 'unknown')}")
    
    # Compute hash for verification
    try:
        core_hash = compute_class_list_hash(core_path)
        print(f"Hash: {core_hash}")
    except Exception as e:
        print(f"Hash: (could not compute - {e})")
        core_hash = None
    
    # Check if working copy exists
    if working_path.exists():
        print(f"\n⚠️  Working copy exists: {working_path}")
        
        # Try to compute hash of existing
        try:
            working_hash = compute_class_list_hash(working_path)
            print(f"   Existing hash: {working_hash}")
            
            if core_hash and working_hash == core_hash:
                print(f"   ✓ Already synchronized!")
                if not force:
                    response = input("\n   Copy anyway? (y/n): ").strip().lower()
                    if response != 'y':
                        print("   Cancelled.")
                        return
        except Exception as e:
            print(f"   Existing hash: (could not compute - {e})")
        
        if not force:
            response = input(f"\n   Overwrite working copy? (y/n): ").strip().lower()
            if response != 'y':
                print("   Cancelled.")
                return
    
    # Copy core to working location
    print(f"\n📋 Copying core → working...")
    print(f"   From: {core_path}")
    print(f"   To: {working_path}")
    
    working_path.parent.mkdir(parents=True, exist_ok=True)
    shutil.copy2(core_path, working_path)
    
    # Verify copy
    if working_path.exists():
        # Ensure working copy is NOT read-only (should be editable by scripts)
        try:
            import os
            os.chmod(working_path, 0o666)  # Make writable
        except:
            pass
        
        print(f"\n✓ Successfully synchronized!")
        print(f"\n{'='*70}")
        print(f"RESULT")
        print(f"{'='*70}")
        print(f"Working data.yaml ready at: {working_path}")
        print(f"Version: v{version}")
        print(f"Status: ✓ Synchronized with core")
        
        if core_hash:
            verify_hash = compute_class_list_hash(working_path)
            if verify_hash == core_hash:
                print(f"Hash: ✓ Verified ({verify_hash})")
            else:
                print(f"Hash: ⚠️ MISMATCH (expected {core_hash}, got {verify_hash})")
    else:
        print(f"❌ ERROR: Copy failed!")
        sys.exit(1)


def main():
    parser = argparse.ArgumentParser(
        description="Sync data.yaml working copy from protected core class definitions",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Sync from v1 (initial 2,641 classes)
  python scripts/class_management/sync_data_yaml.py --version 1
  
  # Force overwrite without confirmation
  python scripts/class_management/sync_data_yaml.py --version 1 --force
  
  # Sync from v2 (after extending with new cards)
  python scripts/class_management/sync_data_yaml.py --version 2
        """
    )
    
    parser.add_argument(
        '--version',
        type=int,
        required=True,
        help='Core version to sync from (1, 2, etc.)'
    )
    
    parser.add_argument(
        '--force',
        action='store_true',
        help='Overwrite without confirmation'
    )
    
    args = parser.parse_args()
    
    sync_data_yaml(args.version, args.force)


if __name__ == '__main__':
    main()
