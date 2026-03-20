# AioIR: Unified All-in-one Image Restoration Framework

## Installation and Data Preparation

See [INSTALL.md](INSTALL.md) for the installation of dependencies and dataset preperation required to run this codebase.

## Training

After preparing the training data in ```data/``` directory, use 
```
python train.py -opt options/train/aioir_base.yml
```
to start the training of the model.

The project now supports a BasicSR-like configuration style:
- model and architecture are managed by registries (see `models/` and `net/`)
- experiment settings are managed by YAML files in `options/`
- define different model base classes and map them in `models/__init__.py`

Network registration supports BasicSR-like usage:
- decorator registration via `@register_arch("YourArchName")`
- automatic module discovery in `net/` for files with `_arch.py` suffix.

Use the `datasets.train.de_type` field in the yml to choose the combination of degradation types to train on.

Example Usage: If we only want to train on deraining and dehazing:
```
python train.py -opt options/train/aioir_base.yml
```
Then set:
```yaml
datasets:
	train:
		de_type: [derain, dehaze]
```

For basic instruction, we offer a [PromptIR](https://github.com/va1shn9v/PromptIR) model and option file in net/PromptIR.py and options/PromptIR.yml, they are adjusted to fit this repository's structure.

## Testing
```
python test.py -opt options/test/aioir_5task.yml
```

Set `test.mode` in yml to one of:
- `denoise`
- `derain`
- `dehaze`
- `deblur`
- `enhance`
- `3task`
- `5task`

Example Usage: To test on all the degradation types at once, run:

```
python test.py -opt options/test/aioir_5task.yml
```

**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR), [AdaIR](https://github.com/c-yn/AdaIR) and [BasicSR](https://github.com/XPixelGroup/BasicSR) repository

