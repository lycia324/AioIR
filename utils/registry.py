ARCH_REGISTRY = {}
DATASET_REGISTRY = {}
MODEL_REGISTRY = {}

def register_arch(name=None):
    def _register(cls):
        arch_name = name or cls.__name__
        if arch_name in ARCH_REGISTRY:
            raise KeyError(f"Architecture '{arch_name}' is already registered")
        ARCH_REGISTRY[arch_name] = cls
        return cls

    return _register

def register_dataset(name=None):
    def _register(cls):
        dataset_name = name or cls.__name__
        if dataset_name in DATASET_REGISTRY:
            raise KeyError(f"Dataset '{dataset_name}' is already registered")
        DATASET_REGISTRY[dataset_name] = cls
        return cls

    return _register

def register_model(name=None):
    def _register(cls):
        model_name = name or cls.__name__
        if model_name in MODEL_REGISTRY:
            raise KeyError(f"Model '{model_name}' is already registered")
        MODEL_REGISTRY[model_name] = cls
        return cls

    return _register

def build_network(opt):
    if "type" not in opt:
        raise KeyError("Network option must contain key: type")

    net_type = opt["type"]
    if net_type not in ARCH_REGISTRY:
        support = ", ".join(sorted(ARCH_REGISTRY.keys()))
        raise ValueError(f"Unknown network type: {net_type}. Available: [{support}]")

    kwargs = {k: v for k, v in opt.items() if k != "type"}
    return ARCH_REGISTRY[net_type](**kwargs)

def build_dataset(opt, **kwargs):
    if "type" not in opt:
        raise KeyError("Dataset option must contain key: type")

    dataset_type = opt["type"]
    if dataset_type not in DATASET_REGISTRY:
        support = ", ".join(sorted(DATASET_REGISTRY.keys()))
        raise ValueError(f"Unknown dataset type: {dataset_type}. Available: [{support}]")

    init_opt = {k: v for k, v in opt.items() if k != "type"}
    return DATASET_REGISTRY[dataset_type](**init_opt, **kwargs)

def build_model(opt, **kwargs):
    if "type" not in opt:
        raise KeyError("Model option must contain key: type")

    model_type = opt["type"]
    if model_type not in MODEL_REGISTRY:
        support = ", ".join(sorted(MODEL_REGISTRY.keys()))
        raise ValueError(f"Unknown model type: {model_type}. Available: [{support}]")

    init_opt = {k: v for k, v in opt.items() if k != "type"}
    return MODEL_REGISTRY[model_type](**init_opt, **kwargs)