"""EfficientNet-B0 transfer learning model builder."""

import torchvision
from torch import nn


def create_efficientnet_b0(num_classes: int, freeze_features: bool = True) -> nn.Module:
    """Load ImageNet pre-trained EfficientNet-B0 and replace the classifier head."""
    weights = torchvision.models.EfficientNet_B0_Weights.DEFAULT
    model = torchvision.models.efficientnet_b0(weights=weights)

    if freeze_features:
        for param in model.features.parameters():
            param.requires_grad = False

    in_features = model.classifier[1].in_features
    model.classifier = nn.Sequential(
        nn.Dropout(p=0.3, inplace=True),
        nn.Linear(in_features=in_features, out_features=num_classes),
    )

    return model
def unfreeze_top_layers(model: nn.Module, num_layers: int = 4) -> nn.Module:
    """Unfreeze the last *num_layers* blocks in the feature extractor for fine-tuning."""
    feature_blocks = list(model.features.children())
    for block in feature_blocks[-num_layers:]:
        for param in block.parameters():
            param.requires_grad = True
    return model


def get_parameter_groups(
    model: nn.Module,
    classifier_lr: float,
    backbone_lr: float,
) -> list[dict]:
    """Return optimizer param groups with separate learning rates for head and backbone."""
    classifier_params = list(model.classifier.parameters())
    classifier_ids = {id(p) for p in classifier_params}
    backbone_params = [
        p for p in model.parameters() if p.requires_grad and id(p) not in classifier_ids
    ]

    return [
        {"params": classifier_params, "lr": classifier_lr},
        {"params": backbone_params, "lr": backbone_lr},
    ]
