# Deprecated Training Script

⚠️ **This script is deprecated and will be removed in a future release.**

Please use the new unified training interface instead:

## Migration

### Old Way
```bash
python core/train.py --dataset all --epochs 10
```

### New Way
```bash
# From repository root:
python train.py --dataset all --epochs 10

# Or use a configuration file:
python train.py --config config/training/quick_test.yaml
```

## Benefits

- Configuration-driven workflows (YAML/JSON)
- Better documentation and examples
- Consistent interface across all datasets
- Easier to extend and maintain

## Documentation

See [Script Consolidation Guide](docs/SCRIPT_CONSOLIDATION.md) for complete documentation.
