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
- automatic module discovery in `net/` for files with `_arch.py` suffix and current `OriSSM_*` modules

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

## Testing

After preparing the testing data in ```test/``` directory, place the model checkpoint file in the ```ckpt``` directory. The pre-trained model can be downloaded [here](https://drive.google.com/drive/folders/1x2LN4kWkO3S65jJlH-1INUFiYt8KFzPH?usp=sharing). To perform the evaluation, use
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
<!-- 
## Demo
To obtain visual results from the model ```demo.py``` can be used. After placing the saved model file in ```ckpt``` directory, run:
```
python demo.py --test_path {path_to_degraded_images} --output_path {save_images_here}
```
Example usage to run inference on a directory of images:
```
python demo.py --test_path './test/demo/' --output_path './output/demo/'
```
Example usage to run inference on an image directly:
```
python demo.py --test_path './test/demo/image.png' --output_path './output/demo/'
```
To use tiling option while running ```demo.py``` set ```--tile``` option to ```True```. The Tile size and Tile overlap parameters can be adjusted using ```--tile_size``` and ```--tile_overlap``` options respectively. -->


**Acknowledgment:** This code is based on the [PromptIR](https://github.com/va1shn9v/PromptIR), [AdaIR](https://github.com/c-yn/AdaIR) and [BasicSR](https://github.com/XPixelGroup/BasicSR) repository

