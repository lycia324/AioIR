from models.aioir_model import AioIRModel


MODEL_REGISTRY = {
    "AioIRModel": AioIRModel,
}


def build_model(opt):
    model_opt = opt.get("model", {})
    model_type = model_opt.get("type", "AioIRModel")
    if model_type not in MODEL_REGISTRY:
        raise ValueError(f"Unknown model type: {model_type}")
    return MODEL_REGISTRY[model_type](opt)
