import torch

from prototypes import pixelwise_multiply


class ModifiedHeadBase(torch.nn.Module):
    def __init__(self, U=None, device="cuda"):
        super().__init__()
        self.U = U
        self.device = device

    @property
    def disentanglement_matrix(self):
        return self.U

    @property
    def fc_weight(self):
        if self.U is None:
            return self.linear_weight
        return self.linear_weight @ self.U.inverse()

    def _preprocess_input(self, x):
        if self.U is None:
            return x
        return pixelwise_multiply(x, self.U())

    def _pool_and_flatten(self, x):
        raise NotImplementedError("Subclasses must implement this method.")

    def forward(self, x, preprocess=True):
        if preprocess:
            x = self._preprocess_input(x)
        x = self._pool_and_flatten(x)
        return torch.nn.functional.linear(x, self.fc_weight, self.b)

    def before_linear(self, x):
        x = self._preprocess_input(x)
        return self._pool_and_flatten(x)


class ModifiedHeadResnet(ModifiedHeadBase):
    def __init__(self, model, U=None, device="cuda"):
        super().__init__(U, device)
        model = model.to(device)
        self.avgpool = list(model.children())[-2]
        fc = list(model.children())[-1]
        self.linear_weight = fc.weight
        self.b = fc.bias

    def _pool_and_flatten(self, x):
        x = self.avgpool(x)
        return torch.flatten(x, 1)


class ModifiedHeadConvNeXt(ModifiedHeadBase):
    def __init__(self, model, U=None, device="cuda"):
        super().__init__(U, device)
        model = model.to(device)
        self.avgpool = list(model.children())[-2]
        classifier = list(model.children())[-1]
        self.eps = model.classifier[0].eps
        self.gamma = model.classifier[0].weight
        self.beta = model.classifier[0].bias
        self.flatten = classifier[1]
        fc = classifier[2]
        self.linear_weight = fc.weight
        self.b = fc.bias

    def _pool_and_flatten(self, x):
        x = self.avgpool(x)
        return self.flatten(x)


class ModifiedHeadDenseNet(ModifiedHeadBase):
    def __init__(self, model, U=None, device="cuda"):
        super().__init__(U, device)
        model = model.to(device)
        fc = model.classifier
        self.linear_weight = fc.weight
        self.b = fc.bias

    def _pool_and_flatten(self, x):
        x = torch.nn.functional.adaptive_avg_pool2d(x, (1, 1))
        return torch.flatten(x, 1)


class ModifiedHeadSwinTransformer(ModifiedHeadBase):
    def __init__(self, model, U=None, device="cuda"):
        super().__init__(U, device)
        model = model.to(device)
        self.avgpool = list(model.children())[-3]
        self.flatten = list(model.children())[-2]
        fc = list(model.children())[-1]
        self.linear_weight = fc.weight
        self.b = fc.bias

    def _pool_and_flatten(self, x):
        x = self.avgpool(x)
        return self.flatten(x)


HEAD_CLASSES = {
    "resnet": ModifiedHeadResnet,
    "convnext": ModifiedHeadConvNeXt,
    "densenet": ModifiedHeadDenseNet,
    "swin": ModifiedHeadSwinTransformer,
}


def create_modified_head(model, model_name, U=None):
    for prefix, cls in HEAD_CLASSES.items():
        if prefix in model_name:
            return cls(model, U)
    raise ValueError(f"No modified head implementation for '{model_name}'")
