"""Verify class ID assignments."""
import yaml

with open('data/synthetic/data.yaml', encoding='utf-8') as f:
    data = yaml.safe_load(f)

print(f"Total classes: {len(data['names'])}")
print(f"\nClass ID assignments:")
print(f"Class 0: {data['names'][0]}")
print(f"Class 1: {data['names'][1]}")
print(f"Class 2: {data['names'][2]}")
print(f"...")
print(f"Class 2638: {data['names'][2638]}")
print(f"Class 2639: {data['names'][2639]}")
print(f"Class 2640: {data['names'][2640]}")
print(f"\n✓ Line numbers in the file are IRRELEVANT")
print(f"✓ Only the ORDER in the list matters")
