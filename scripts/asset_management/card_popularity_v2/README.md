# Card Popularity V2 - Complete Overhaul

# Card Popularity V2 System

**Status: ✅ ACTIVE - Production Ready**

Multi-format card popularity scraping system for Flesh and Blood synthetic data generation.

## Goals

This is a complete rewrite of the card popularity scraping and weighting system with the following improvements:

### Key Improvements Over V1

1. **Multi-Format Support**
   - Classic Constructed (CC)
   - Living Legend (LL)
   - Blitz
   - Comprehensive format coverage

2. **Robust Data Handling**
   - Better error handling and validation
   - Graceful degradation for missing data
   - Data integrity checks
   - Automatic retry logic

3. **Hero Coverage**
   - Automatic discovery of all heroes per format
   - No hardcoded hero lists
   - Placeholder generation for heroes without deck data
   - Format-specific hero legality

4. **Better Data Structure**
   - Format-aware weighting
   - Separate weights per format
   - Metadata tracking (scrape date, total decks, etc.)
   - Validation and completeness metrics

## Architecture

```
card_popularity_v2/
├── README.md                          # This file
├── scrape_popularity.py              # Main scraper (all formats)
├── validate_weights.py               # Validation and integrity checks
├── merge_formats.py                  # Combine format data intelligently
└── generate_placeholders.py          # Auto-generate missing heroes
```

## Output Structure

```json
{
  "metadata": {
    "version": "2.0",
    "generated": "2025-10-21T12:00:00Z",
    "formats_included": ["cc", "ll", "blitz"],
    "total_heroes": 150,
    "total_cards_tracked": 5000
  },
  "formats": {
    "cc": {
      "total_decks": 50000,
      "heroes": {
        "hero-name": {
          "deck_count": 1000,
          "deck_percentage": 2.0,
          "sections": {
            "equipment": [...],
            "weapon": [...],
            "maindeck": [...]
          }
        }
      }
    },
    "ll": { ... },
    "blitz": { ... }
  }
}
```

## Development Workflow

1. **Phase 1:** Build new scraper with multi-format support
2. **Phase 2:** Create validation tools
3. **Phase 3:** Generate test data and verify against V1
4. **Phase 4:** Update synthetic generation to use V2 format
5. **Phase 5:** Delete V1 scripts and migrate data

## Testing

Before replacing V1 system:
- [ ] Scrape all three formats successfully
- [ ] Validate data integrity
- [ ] Compare CC data with V1 for consistency
- [ ] Test synthetic generation with V2 data
- [ ] Verify all heroes have data (real or placeholder)

## Migration Checklist

- [ ] New scripts fully functional
- [ ] Data validated
- [ ] Synthetic generation updated
- [ ] Documentation updated
- [ ] V1 scripts archived/deleted
- [ ] V1 data backed up then replaced

---

**Note:** This folder will be deleted after successful migration to V2.
