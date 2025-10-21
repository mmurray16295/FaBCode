# Class Management Scripts

This directory contains scripts for managing YOLO class definitions in a version-safe manner.

## Scripts

### `sync_data_yaml.py`
Synchronize working data.yaml from protected core versions.

**Usage:**
```bash
# Sync from v1 (Phase 3 model - 2,641 classes)
python scripts/class_management/sync_data_yaml.py --version 1

# Force overwrite without confirmation
python scripts/class_management/sync_data_yaml.py --version 1 --force
```

**Purpose:** Safely regenerate `data/synthetic/data.yaml` from the read-only core version.

---

### `extend_class_definitions.py`
Extend class definitions with new cards while preserving existing order.

**Usage:**
```bash
# Preview changes (recommended first step)
python scripts/class_management/extend_class_definitions.py --dry-run

# Create new version
python scripts/class_management/extend_class_definitions.py
```

**What it does:**
1. Finds latest core_classes_v*.yaml (e.g., v1)
2. Loads existing class order
3. Finds new cards in card.json
4. Creates v2 with preserved v1 order + new cards appended
5. Makes v2 read-only
6. Instructs you to commit to git

**Example output:**
```
Latest core version: v1 (2,641 classes)
New cards found: 50
Creating: core_classes_v2.yaml (2,691 classes)
  Classes [0-2640]: Preserved from v1
  Classes [2641-2690]: New cards appended
```

---

## Workflow

### Initial Setup (Already Done)
```bash
# Core v1 created from existing data.yaml
# Protected as read-only
# Ready for git tracking
```

### Adding New Cards
```bash
# 1. Update card database
python scripts/download_card_json.py

# 2. Preview new cards
python scripts/class_management/extend_class_definitions.py --dry-run

# 3. Create new core version
python scripts/class_management/extend_class_definitions.py

# 4. Commit to git
git add data/class_definitions/core_classes_v2.yaml
git commit -m "Add core_classes_v2.yaml (50 new cards)"

# 5. Sync working copy
python scripts/class_management/sync_data_yaml.py --version 2

# 6. Continue training
python scripts/train_yolo11x.py --resume Phase3_yolo11x.pt
```

### Emergency: Restore Corrupted data.yaml
```bash
# Restore from core v1
python scripts/class_management/sync_data_yaml.py --version 1 --force
```

---

## Safety Features

1. **Read-Only Protection**: Core versions are filesystem read-only
2. **Git Tracking**: All core versions committed and versioned
3. **Hash Verification**: Scripts compute hashes to verify integrity
4. **Metadata Headers**: Each version documents date, model, and class count
5. **Order Preservation**: New cards always appended, never inserted

---

## Directory Structure

```
data/
├── class_definitions/          # Protected core versions
│   ├── README.md              # Documentation
│   ├── core_classes_v1.yaml   # Phase 3 model (2,641 classes) [READ-ONLY]
│   └── core_classes_v2.yaml   # Future (when new cards added) [READ-ONLY]
│
└── synthetic/
    └── data.yaml              # Working copy (regenerated from core)
```

---

## Warning

**DO NOT:**
- ❌ Edit core_classes_*.yaml files manually
- ❌ Remove read-only protection from core files
- ❌ Run generate_card_data_yaml.py (obsolete - use these scripts)
- ❌ Delete core versions

**DO:**
- ✅ Use these management scripts
- ✅ Commit all core versions to git
- ✅ Document which models use which versions
- ✅ Sync working copy after changing versions
