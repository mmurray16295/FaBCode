# Class Definitions - Protected YOLO Class Order

## ⚠️ CRITICAL: DO NOT MODIFY CORE FILES

This directory contains **version-locked class definitions** for YOLO model training. The class order is **permanently tied** to trained model weights.

## Files

### Core Class Definitions (READ-ONLY)
- **`core_classes_v1.yaml`** - Initial 2,641 classes (Phase 3 model)
  - Date: October 21, 2024
  - Cards: 2,641 unique card names
  - Model: Phase3_yolo11x.pt
  - **Status: READ-ONLY, git tracked**
  
- **`core_classes_v2.yaml`** - Extended version (when new cards added)
  - Will be created by `extend_class_definitions.py`
  - Preserves v1 order + appends new cards
  - **Status: Not yet created**

### Working Copy
- **`data/synthetic/data.yaml`** - Generated from core version
  - Used by training and generation scripts
  - Regenerated via `sync_data_yaml.py`
  - Safe to delete and regenerate

## Why This Matters

YOLO models **bake class indices into weights**:
```
Class 0 → "10,000 Year Reunion" → specific neurons
Class 13 → "Aether Ashwing" → specific neurons
```

**If class order changes:**
- Model thinks Class 13 is a different card
- ALL predictions become wrong
- Must retrain from scratch (expensive!)

## Workflow

### Initial Setup (DONE)
```bash
python scripts/class_management/generate_initial_classes.py
# Creates core_classes_v1.yaml
# Makes it read-only
# Commits to git
```

### Adding New Cards
```bash
# 1. Update card.json with new cards
python scripts/download_card_json.py

# 2. Extend class definitions (preserves order!)
python scripts/class_management/extend_class_definitions.py
# Creates core_classes_v2.yaml
# Appends new cards to END of list
# Makes it read-only

# 3. Sync working copy
python scripts/class_management/sync_data_yaml.py --version 2
# Copies core_classes_v2.yaml → data/synthetic/data.yaml

# 4. Continue training with new classes
python scripts/train_yolo11x.py --resume Phase3_yolo11x.pt
```

### Regenerating Working Copy
```bash
# Safe - just copies from core version
python scripts/class_management/sync_data_yaml.py --version 1
```

## Safeguards

1. **Read-Only Files**: Core versions are read-only on filesystem
2. **Git Tracking**: All core versions tracked in git
3. **Version Headers**: Each file has version metadata
4. **Hash Validation**: Scripts verify class order hasn't changed
5. **No Direct Editing**: Use management scripts only

## Version History

### v1 - October 21, 2024
- Initial 2,641 classes
- Alphabetically sorted from card.json
- Trained: Phase3_yolo11x.pt
- Hash: (to be computed)

### v2 - (Future)
- Preserves v1 classes [0-2640]
- Appends new cards [2641+]
- Trained: (future model)

## Emergency Recovery

If working data.yaml gets corrupted:
```bash
# Restore from core version
python scripts/class_management/sync_data_yaml.py --version 1 --force
```

If core version accidentally modified:
```bash
# Restore from git
git checkout data/class_definitions/core_classes_v1.yaml

# Re-apply read-only protection
Set-ItemProperty "data\class_definitions\core_classes_v1.yaml" -Name IsReadOnly -Value $true
```

## DO NOT

- ❌ Edit core_classes_*.yaml files manually
- ❌ Regenerate data.yaml from card.json without using management scripts
- ❌ Change class order in existing core versions
- ❌ Delete core versions
- ❌ Remove read-only protection

## DO

- ✅ Use `extend_class_definitions.py` to add new cards
- ✅ Use `sync_data_yaml.py` to regenerate working copy
- ✅ Commit core versions to git
- ✅ Document model → version mapping
- ✅ Validate class order before training
