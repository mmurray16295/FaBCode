"""Validate reformatted YAML files."""
import yaml
from pathlib import Path

files = [
    'scripts/class_management/core_classes_v1.yaml',
    'data/synthetic/data.yaml'
]

print("Validating reformatted YAML files...\n")

for filepath in files:
    path = Path(filepath)
    if not path.exists():
        print(f"⚠️  {filepath}: Not found")
        continue
    
    try:
        with open(path, 'r', encoding='utf-8') as f:
            data = yaml.safe_load(f)
        
        print(f"✓ {filepath}")
        print(f"  Classes: {len(data['names'])}")
        print(f"  First: {data['names'][0]}")
        print(f"  Last: {data['names'][-1]}")
        print()
        
    except Exception as e:
        print(f"❌ {filepath}: {e}\n")

print("Validation complete!")
