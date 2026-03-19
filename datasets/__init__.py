from utils.dataset_utils import AdaIRTrainDataset, DenoiseTestDataset, DerainDehazeDataset
from utils.config import dict_to_namespace


DATASET_REGISTRY = {
    "AdaIRTrainDataset": AdaIRTrainDataset,
    "DenoiseTestDataset": DenoiseTestDataset,
    "DerainDehazeDataset": DerainDehazeDataset,
}


def build_dataset(dataset_opt, **kwargs):
    if "type" not in dataset_opt:
        raise KeyError("Dataset option must contain key: type")
    dataset_type = dataset_opt["type"]
    if dataset_type not in DATASET_REGISTRY:
        raise ValueError(f"Unknown dataset type: {dataset_type}")
    cls = DATASET_REGISTRY[dataset_type]
    init_opt = {k: v for k, v in dataset_opt.items() if k != "type"}
    args = dict_to_namespace(init_opt)
    return cls(args, **kwargs)
