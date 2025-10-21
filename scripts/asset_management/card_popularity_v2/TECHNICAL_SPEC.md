# Card Popularity V2 - Technical Specification

**Status: ✅ PRODUCTION - V1 System Fully Replaced**

## Migration Completed (October 21, 2025)

### ✅ Completed Tasks:
1. ✅ Removed Living Legend format (redundant with CC)
2. ✅ Created heroes_card.json with format legality
3. ✅ Built multi-format scraper (CC + Blitz)
4. ✅ Successfully scraped 120 heroes (53 CC, 67 Blitz)
5. ✅ Generated card_weights_all_printings.json (2.2 MB)
6. ✅ Updated all scripts to use new weights file
7. ✅ Removed V1 scripts and old data files
8. ✅ Updated CardSelector for new data structure
9. ✅ Tested and validated synthetic generation

### Final Statistics:
- **CC Format**: 53 adult heroes, 529,944 decks scraped
- **Blitz Format**: 67 young heroes (2 failed: Bravo Flattering Showman, Kayo Strong-arm), 183,024 decks scraped
- **Total Heroes**: 120 unique heroes across both formats
- **Output File**: `data/card_weights_all_printings.json` (2,224.6 KB)

---

## Previous V1 System (Deprecated)

### Problems Fixed:
1. ✅ **Single Format Only** - Now supports CC and Blitz
2. ✅ **Hardcoded Heroes** - Auto-discovery from heroes_card.json
3. ✅ **Manual Gaps** - Failed heroes tracked in metadata
4. ✅ **No Validation** - Built-in format legality validation
5. ✅ **Fragile Pipeline** - Single unified scraper
6. ✅ **Poor Error Handling** - Graceful failure with reporting

### V1 Files (Deleted):
- ❌ `scrape_card_popularity.py` - Replaced by V2 scraper
- ❌ `add_missing_heroes.py` - No longer needed
- ❌ `add_missing_heroes_to_weights.py` - No longer needed
- ❌ `data/card_popularity_weights_by_hero.json` - Replaced by card_weights_all_printings.json

---

## V2 System Design

### Core Principles:
1. ✅ **Format-First Architecture** - CC and Blitz are independent
2. ✅ **Auto-Discovery** - Heroes loaded from heroes_card.json
3. ✅ **Graceful Degradation** - Failed heroes tracked in metadata
4. ✅ **Validation Built-In** - Format legality checked during scraping
5. ✅ **Single Command** - One script does it all

### Active Components:

#### 1. ✅ `extract_heroes.py` (Completed)
**Purpose:** Extract heroes from card.json into dedicated file

**Output:** `data/heroes_card.json`
- 62 adult heroes (CC legal)
- 72 young heroes (Blitz legal)
- Format legality flags for each hero

#### 2. ✅ `scrape_popularity.py` (Active)
**Purpose:** Single entry point for all scraping

**Features:**
- Multi-format support: CC, Blitz (LL removed as redundant)
- Auto-discovers heroes from heroes_card.json
- Scrapes hero deck data with retry logic
- Handles rate limiting (1 sec between heroes)
- Tracks failed heroes in metadata
- Direct output to data/card_weights_all_printings.json

**Usage:**
```bash
# Scrape all formats (default)
python scrape_popularity.py

# Scrape specific format
python scrape_popularity.py --formats cc

# Custom output name
python scrape_popularity.py --output custom_weights.json
```

**Output:** `data/card_weights_all_printings.json`

#### 3. ✅ `test_scraper.py` (Active)
**Purpose:** Test individual heroes before full scrape

**Usage:**
```bash
python test_scraper.py "Dorinthea Ironsong" cc
python test_scraper.py "Pleiades, Superstar" cc
```

**Output:** `test_output.json` (for validation)

---

## Deferred/Future Components

The following components were planned but are not currently needed:

#### `validate_weights.py` (Not Implemented)
- Original plan: Validate scraped data integrity
- Status: Not needed - validation happens during scrape
- Future: Could add post-scrape analysis reports

#### `merge_formats.py` (Not Implemented)
- Original plan: Merge CC/LL/Blitz into unified file
- Status: Scraper already outputs unified structure
- Formats are in `formats.cc` and `formats.blitz` keys

#### `generate_placeholders.py` (Not Implemented)
- Original plan: Generate placeholder data for missing heroes
- Status: Not needed - synthetic generation handles missing heroes gracefully
- CardSelector falls back to unweighted selection for heroes without weights
```bash
python generate_placeholders.py --format cc
python generate_placeholders.py --all
```

---

## Data Structure V2

### Format-Specific Weight File: `weights_{format}.json`

```json
{
  "metadata": {
    "format": "cc",
    "version": "2.0",
    "generated": "2025-10-21T12:34:56Z",
    "total_decks": 50000,
    "total_heroes": 45,
    "heroes_with_data": 40,
    "heroes_placeholder": 5,
    "source": "fabrec.gg",
    "scraper_version": "2.0.0"
  },
  "heroes": {
    "hero-name": {
      "name_display": "Hero Name",
      "deck_count": 1000,
      "deck_percentage": 2.0,
      "data_source": "scraped",  // or "placeholder"
      "total_unique_cards": 45,
      "sections": {
        "equipment": [
          {
            "card_name": "Fyendal's Spring Tunic",
            "card_id": "ARC151",
            "usage_percentage": 95.5,
            "usage_count": 955
          }
        ],
        "weapon": [...],
        "maindeck": [...]
      }
    }
  }
}
```

### Unified Weight File: `weights_unified.json`

```json
{
  "metadata": {
    "version": "2.0",
    "generated": "2025-10-21T12:45:00Z",
    "formats_included": ["cc", "ll", "blitz"],
    "total_heroes_cc": 45,
    "total_heroes_ll": 12,
    "total_heroes_blitz": 60
  },
  "formats": {
    "cc": { /* full cc data */ },
    "ll": { /* full ll data */ },
    "blitz": { /* full blitz data */ }
  }
}
```

---

## Implementation Plan

### Phase 1: Core Scraper
- [ ] Build `scrape_popularity.py` foundation
- [ ] Implement format auto-discovery
- [ ] Add hero list scraping per format
- [ ] Add individual hero page scraping
- [ ] Add card.json validation
- [ ] Test with CC format first

### Phase 2: Placeholder Generation
- [ ] Query card.json for all legal heroes
- [ ] Implement smart placeholder logic
- [ ] Add class/talent-aware generic cards
- [ ] Integrate into main scraper

### Phase 3: Validation
- [ ] Build `validate_weights.py`
- [ ] Add all integrity checks
- [ ] Create validation report format
- [ ] Test against scraped data

### Phase 4: Format Merging
- [ ] Build `merge_formats.py`
- [ ] Test with all three formats
- [ ] Verify unified structure

### Phase 5: Integration
- [ ] Update `card_selector.py` to use V2 format
- [ ] Add format selection logic
- [ ] Test synthetic generation
- [ ] Performance testing

### Phase 6: Migration
- [ ] Back up V1 data
- [ ] Replace V1 files with V2
- [ ] Update all documentation
- [ ] Delete V1 scripts

---

## Testing Strategy

### Unit Tests:
- Card name normalization
- Hero key generation
- Percentage calculations
- Data structure validation

### Integration Tests:
- Full scrape of CC format
- Compare with V1 CC data (should be similar)
- Scrape LL and Blitz formats
- Validate merged output

### Production Tests:
- Generate 1000 synthetic images with V2 data
- Verify hero distribution matches expectations
- Check card selection quality
- Compare with V1 generation

---

## Success Criteria

1. ✅ All three formats scraped successfully
2. ✅ All heroes have data (real or placeholder)
3. ✅ Validation passes 100%
4. ✅ Synthetic generation works with V2 data
5. ✅ Performance is equal or better than V1
6. ✅ Documentation is complete

---

## Timeline Estimate

- **Phase 1:** 2-3 hours (core scraper)
- **Phase 2:** 1 hour (placeholders)
- **Phase 3:** 1 hour (validation)
- **Phase 4:** 30 minutes (merging)
- **Phase 5:** 1-2 hours (integration)
- **Phase 6:** 30 minutes (migration)

**Total:** ~6-8 hours of development + testing

---

## Notes

- Keep V1 system running during development
- Test each phase thoroughly before proceeding
- Document any issues or edge cases discovered
- Consider caching scraped data to avoid re-scraping during testing
