import os
import yaml
from collections import defaultdict
import re

# Scans a YOLO dataset root (with train/valid/test subfolders) and builds per-class counts
# across splits based on labels, aligned to a provided classes.yaml (names list),
# then writes coverage.yaml with structure:
#   classes: [name1, name2, ...]
#   counts:
#     train: { name1: 10, name2: 0, ... }
#     valid: { ... }
#     test:  { ... }
#     total: { ... }
# Also prints a short summary with number of missing classes per split.

def canonicalize_name(name: str) -> str:
    m = re.match(r"^(.*)_([A-Z]{2,5}\d{1,5})$", name)
    return m.group(1) if m else name


def load_class_names(classes_yaml_path):
    with open(classes_yaml_path, 'r') as f:
        data = yaml.safe_load(f) or {}
    names = data.get('names')
    if not isinstance(names, list):
        raise ValueError('classes.yaml missing names list')
    # Canonicalize and dedupe preserving order
    canon = []
    seen = set()
    for n in names:
        cn = canonicalize_name(n)
        if cn not in seen:
            canon.append(cn)
            seen.add(cn)
    return canon


def discover_set_classes(images_root):
    union = set()
    for set_name in ['SEA', 'WTR', 'HVY']:
        set_dir = os.path.join(images_root, set_name)
        if not os.path.isdir(set_dir):
            continue
        for fn in os.listdir(set_dir):
            if not fn.lower().endswith('.png'):
                continue
            base = os.path.splitext(fn)[0]
            union.add(canonicalize_name(base))
    return sorted(union)


def init_counts(names):
    return {n: 0 for n in names}


def scan_labels_for_split(split_dir, names):
    counts = init_counts(names)
    label_dir = os.path.join(split_dir, 'labels')
    if not os.path.isdir(label_dir):
        return counts
    for fn in os.listdir(label_dir):
        if not fn.lower().endswith('.txt'):
            continue
        path = os.path.join(label_dir, fn)
        try:
            with open(path, 'r') as f:
                for line in f:
                    parts = line.strip().split()
                    if len(parts) != 5:
                        continue
                    cls_id = int(parts[0])
                    if 0 <= cls_id < len(names):
                        counts[names[cls_id]] += 1
        except Exception:
            # Skip malformed files
            pass
    return counts


def add_totals(counts):
    total = {}
    for name in counts['train']:
        total[name] = counts['train'][name] + counts['valid'][name] + counts['test'][name]
    counts['total'] = total


def write_coverage_yaml(out_path, names, counts):
    data = {
        'classes': names,
        'counts': counts
    }
    os.makedirs(os.path.dirname(out_path), exist_ok=True)
    with open(out_path, 'w') as f:
        yaml.dump(data, f, default_flow_style=False)


def main():
    # Defaults aligned to repo structure; override via env vars if needed
    root = os.environ.get('FAB_SYNTH_ROOT', r'C:\VS Code\FaB Code\data\synthetic 2')
    classes_yaml = os.path.join(root, 'classes.yaml')
    coverage_yaml = os.path.join(root, 'coverage.yaml')
    images_root = os.path.join(os.path.dirname(root), 'images')

    names_existing = load_class_names(classes_yaml)
    names_discovered = discover_set_classes(images_root)
    # Union: keep existing order, then append any discovered not already present
    names = []
    seen = set()
    for n in names_existing + [n for n in names_discovered if n not in names_existing]:
        if n not in seen:
            names.append(n)
            seen.add(n)

    counts = {
        'train': scan_labels_for_split(os.path.join(root, 'train'), names),
        'valid': scan_labels_for_split(os.path.join(root, 'valid'), names),
        'test': scan_labels_for_split(os.path.join(root, 'test'), names),
    }
    add_totals(counts)

    write_coverage_yaml(coverage_yaml, names, counts)

    # Summary
    def missing(stats):
        return sum(1 for v in stats.values() if v == 0)
    print(f"Coverage written to {coverage_yaml}")
    print(f"Missing classes -> train: {missing(counts['train'])}, valid: {missing(counts['valid'])}, test: {missing(counts['test'])}, total: {missing(counts['total'])}")


if __name__ == '__main__':
    main()
