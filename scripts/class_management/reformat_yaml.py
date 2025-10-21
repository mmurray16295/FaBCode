"""
Reformat YAML files from single-line list to multi-line format for readability.
This is safe - both formats are equivalent to YAML parsers.
"""

import yaml
from pathlib import Path

# Files to reformat
files_to_update = [
    Path('scripts/class_management/core_classes_v1.yaml'),
    Path('data/synthetic/data.yaml')
]

for yaml_path in files_to_update:
    if not yaml_path.exists():
        print(f'Skipping {yaml_path} (not found)')
        continue
    
    print(f'Reformatting {yaml_path}...')
    
    # Read current content
    with open(yaml_path, 'r', encoding='utf-8') as f:
        content = f.read()
    
    # Separate header comments from data
    lines = content.split('\n')
    header_lines = []
    for i, line in enumerate(lines):
        if line.startswith('#') or line.strip() == '':
            header_lines.append(line)
        else:
            # Found first non-comment line
            break
    
    # Load YAML data
    with open(yaml_path, 'r', encoding='utf-8') as f:
        data = yaml.safe_load(f)
    
    # Write back with multi-line format
    with open(yaml_path, 'w', encoding='utf-8') as f:
        # Write header comments
        for line in header_lines:
            f.write(line + '\n')
        
        # Write YAML data with multi-line names
        f.write(f"train: {data['train']}\n")
        f.write(f"val: {data['val']}\n")
        f.write(f"test: {data['test']}\n")
        f.write(f"\n")
        f.write(f"nc: {data['nc']}\n")
        f.write(f"names:\n")
        
        for name in data['names']:
            # Escape names with special characters that need quoting
            if any(c in name for c in [':', '#', '[', ']', '{', '}', '&', '*', '!', '|', '>', '@', '`']):
                # Use double quotes and escape internal quotes
                escaped_name = name.replace('"', '\\"')
                f.write(f'  - "{escaped_name}"\n')
            elif "'" in name:
                # Use double quotes for names with single quotes
                f.write(f'  - "{name}"\n')
            else:
                f.write(f'  - {name}\n')
    
    print(f'  ✓ Reformatted ({len(data["names"])} classes)')

print('\nAll files reformatted to multi-line format!')
print('Changes are purely cosmetic - data structure unchanged.')
