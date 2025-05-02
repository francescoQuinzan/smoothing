# Import necessary libraries
import torch
import torchvision.transforms as transforms

from datasets import _CIFAR10_MEAN, _CIFAR10_STDDEV, _IMAGENET_MEAN, _IMAGENET_STDDEV

DEVICE = torch.device("cuda")

# This line modifies the default SSL context used for HTTPS requests to ensure compatibility with servers.
import ssl
ssl._create_default_https_context = ssl._create_stdlib_context

# List of supported model architectures
CIFAR_ARCHITECTURES = [
    "resnet20",
    "resnet32",
    "resnet44",
    "resnet56",
    "vgg11_bn",
    "vgg13_bn",
    "vgg16_bn",
    "vgg19_bn",
    "mobilenetv2_x1_0",
    "mobilenetv2_x0_75",
    "mobilenetv2_x0_5",
    "mobilenetv2_x1_4",
    "shufflenetv2_x1_0",
    "shufflenetv2_x0_5",
    "shufflenetv2_x1_5",
    "shufflenetv2_x2_0",
    "repvgg_a0",
    "repvgg_a1",
    "repvgg_a2",
    "VAE"
]

# List of supported model IMAGENET names
IMAGENET_ARCHITECTURES = [
    "resnet18", "resnet34", "resnet50", "resnet101", "resnet152",
    "wide_resnet50_2", "wide_resnet101_2",
    "resnext50_32x4d", "resnext101_32x8d",
    "vgg11", "vgg11_bn", "vgg13", "vgg13_bn", "vgg16", "vgg16_bn", "vgg19", "vgg19_bn",
    "densenet121", "densenet169", "densenet201", "densenet161",
    "inception_v3",
    "googlenet",
    "shufflenet_v2_x0_5", "shufflenet_v2_x1_0", "shufflenet_v2_x1_5", "shufflenet_v2_x2_0",
    "mobilenet_v2", "mobilenet_v3_large", "mobilenet_v3_small",
    "mnasnet0_5", "mnasnet0_75", "mnasnet1_0", "mnasnet1_3",
    "efficientnet_b0", "efficientnet_b1", "efficientnet_b2", "efficientnet_b3",
    "efficientnet_b4", "efficientnet_b5", "efficientnet_b6", "efficientnet_b7",
    "regnet_y_400mf", "regnet_y_800mf", "regnet_y_1_6gf", "regnet_y_3_2gf",
    "regnet_y_8gf", "regnet_y_16gf", "regnet_y_32gf", "regnet_x_400mf",
    "regnet_x_800mf", "regnet_x_1_6gf", "regnet_x_3_2gf", "regnet_x_8gf",
    "regnet_x_16gf", "regnet_x_32gf",
    "squeezenet1_0", "squeezenet1_1",
    "alexnet",
    "vit_b_16", "vit_b_32", "vit_l_16", "vit_l_32",
    "convnext_tiny", "convnext_small", "convnext_base", "convnext_large"
]

def load_cifar10_model(model_name: str):
    """
    Loads a pre-trained CIFAR-10 model from the chenyaofo/pytorch-cifar-models repository.

    Parameters:
    - model_name: The name of the model to load.

    Returns:
    - The pre-trained model.
    """
    
    # Mapping from simple model names to full CIFAR-10 model names
    if model_name == "resnet20":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet20', pretrained=True)
    elif model_name == "resnet32":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet32', pretrained=True)
    elif model_name == "resnet44":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet44', pretrained=True)
    elif model_name == "resnet56":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_resnet56', pretrained=True)
    elif model_name == "vgg11_bn":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg11_bn', pretrained=True)
    elif model_name == "vgg13_bn":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg13_bn', pretrained=True)
    elif model_name == "vgg16_bn":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg16_bn', pretrained=True)
    elif model_name == "vgg19_bn":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_vgg19_bn', pretrained=True)
    elif model_name == "mobilenetv2_x1_0":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_mobilenetv2_x1_0', pretrained=True)
    elif model_name == "mobilenetv2_x0_75":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_mobilenetv2_x0_75', pretrained=True)
    elif model_name == "mobilenetv2_x0_5":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_mobilenetv2_x0_5', pretrained=True)
    elif model_name == "mobilenetv2_x1_4":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_mobilenetv2_x1_4', pretrained=True)
    elif model_name == "shufflenetv2_x1_0":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_shufflenetv2_x1_0', pretrained=True)
    elif model_name == "shufflenetv2_x1_5":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_shufflenetv2_x1_5', pretrained=True)
    elif model_name == "shufflenetv2_x0_5":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_shufflenetv2_x0_5', pretrained=True)
    elif model_name == "shufflenetv2_x2_0":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_shufflenetv2_x2_0', pretrained=True)
    elif model_name == "repvgg_a0":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_repvgg_a0', pretrained=True)
    elif model_name == "repvgg_a1":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_repvgg_a1', pretrained=True)
    elif model_name == "repvgg_a2":
        model = torch.hub.load('chenyaofo/pytorch-cifar-models', 'cifar10_repvgg_a2', pretrained=True)
    else:
        raise ValueError(f"Unsupported model '{model_name}' for CIFAR-10. Please choose a valid CIFAR-10 model.")

    return model


def load_imagenet_model(model_name: str):
    """
    Loads a pre-trained ImageNet model from the pytorch/vision repository.

    Parameters:
    - model_name: The name of the model to load.

    Returns:
    - The pre-trained model.
    """

    # Check if the model name is supported and load the model
    if model_name in IMAGENET_ARCHITECTURES:
        model = torch.hub.load('pytorch/vision:v0.10.0', model_name, pretrained=True)
    else:
        raise ValueError(f"Unsupported model '{model_name}' for ImageNet. Please choose a valid ImageNet model.")

    return model
    


def get_normalize_layer(dataset):
    """
    Returns a normalization layer for the given dataset.

    Parameters:
    - dataset: The name of the dataset for which to create the normalization layer.

    Returns:
    - A torchvision.transforms.Normalize layer specific to the dataset.
    """
    # Convert dataset name to lowercase
    dataset = dataset.lower()

    # Define normalization parameters for CIFAR-10 dataset
    if dataset == 'cifar10':
        normalize = transforms.Normalize(mean=_CIFAR10_MEAN, std=_CIFAR10_STDDEV)
    # Define normalization parameters for ImageNet dataset
    elif dataset == 'imagenet':
        normalize = transforms.Normalize(mean=_IMAGENET_MEAN, std=_IMAGENET_STDDEV)
    else:
        raise ValueError(f"Unsupported dataset: {dataset}")
    
    return normalize



def get_architecture(arch: str, dataset: str) -> torch.nn.Module:
    """
    Returns a neural network model with a normalization layer for the given architecture and dataset.

    Parameters:
    - arch: The architecture of the model to load. It should be in the ARCHITECTURES list.
    - dataset: The dataset name, should be supported by the model.

    Returns:
    - A PyTorch nn.Module combining the normalization layer and the model.
    """

    if dataset == 'cifar10':
        model = load_cifar10_model(arch)
        normalize_layer = get_normalize_layer(dataset)
        model = torch.nn.Sequential(normalize_layer, model)

    elif dataset == 'imagenet':
        model = load_imagenet_model(arch)
        normalize_layer = get_normalize_layer(dataset)
        model = torch.nn.Sequential(normalize_layer, model)
    
    model.to(DEVICE)
    model.eval()
    return model
