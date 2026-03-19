from utils.config import dict_to_namespace
from utils.dataset_utils import AdaIRTrainDataset, DenoiseTestDataset, DerainDehazeDataset
from utils.registry import register_dataset


@register_dataset("AdaIRTrainDataset")
class RegisteredAdaIRTrainDataset(AdaIRTrainDataset):
    def __init__(self, **kwargs):
        args = dict_to_namespace(kwargs)
        super().__init__(args)


@register_dataset("DenoiseTestDataset")
class RegisteredDenoiseTestDataset(DenoiseTestDataset):
    def __init__(self, **kwargs):
        args = dict_to_namespace(kwargs)
        super().__init__(args)


@register_dataset("DerainDehazeDataset")
class RegisteredDerainDehazeDataset(DerainDehazeDataset):
    def __init__(self, addnoise=False, sigma=None, task="derain", **kwargs):
        args = dict_to_namespace(kwargs)
        super().__init__(args, task=task, addnoise=addnoise, sigma=sigma)
