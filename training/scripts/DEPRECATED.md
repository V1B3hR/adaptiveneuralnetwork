# Deprecated Training Scripts

⚠️ **These scripts are deprecated and will be removed in a future release.**

Please use the new unified training interface instead:

## Migration Guide

### Old Way
```bash
python training/scripts/train_new_datasets.py --dataset vr_driving --epochs 10
python training/scripts/train_kaggle_datasets.py --dataset annomi --data-path data/
python training/scripts/train_annomi.py --data-path data/annomi
```

### New Way
```bash
# From repository root:
python train.py --dataset vr_driving --epochs 10
python train.py --dataset annomi --data-path data/
python train.py --dataset annomi --data-path data/annomi

# Or with configuration files:
python train.py --config config/training/kaggle_default.yaml
```

## Benefits of the New Interface

1. **Configuration-Driven**: Use YAML/JSON files for reproducible experiments
2. **Consistent CLI**: Same interface for all datasets
3. **Better Defaults**: Sensible default configurations
4. **Extensible**: Easy to add new datasets and models
5. **Well-Documented**: Comprehensive documentation and examples

## Documentation

For complete documentation, see:
- [Script Consolidation Guide](../../docs/SCRIPT_CONSOLIDATION.md)
- [Quick Start Guide](../../QUICKSTART.md)
- [Main README](../../README.md)

## Timeline

- **Now**: Old scripts still work but are deprecated
- **Next minor release**: Deprecation warnings added
- **Next major release**: Old scripts removed

Please update your workflows to use the new unified interface.
