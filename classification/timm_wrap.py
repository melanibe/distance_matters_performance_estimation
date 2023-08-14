import timm


class TimmModelWrapper:
    """
    Wrapper for timm models to match our inference API.
    """

    def __init__(self, model_name: str) -> None:
        self.model = timm.create_model(model_name, pretrained=True)
        self.model.eval()
        self.transform = timm.data.create_transform(**timm.data.resolve_data_config(self.model.pretrained_cfg))
        self.model_name = model_name
        self.model_without_classifier = timm.create_model(model_name, pretrained=True, num_classes=0)
        self.model_without_classifier.eval()
