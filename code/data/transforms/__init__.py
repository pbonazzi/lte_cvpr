import torchvision.transforms as T
from .lgt_transforms import ActiveSpatialCrop, BinaryEmbedding
from .event_augmentations import (
    Denoise,
    EventTransformCompose,
    RandomFlipLR,
    SpatialJitter,
    TensorEventTransformCompose,
    TensorRandomFlipLR,
    TensorSpatialJitter,
)

DATASET_TRANSFORM = {
    "CIFAR-10":
        {
            "train":T.Compose([ 
                        T.RandomHorizontalFlip(),              
                        T.ColorJitter(
                            brightness=0.2, 
                            contrast=0.2, 
                            saturation=0.2, 
                            hue=0.05
                        ),    
                        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1)),  
                        BinaryEmbedding() 
                    ]), 
            
            "val":BinaryEmbedding()  
        },
    "N-Cars":
        {
            "train":T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
                    ]),

            "val": None
        },
    "N-Caltech101":
        {
            "train":T.Compose([
                        T.RandomHorizontalFlip(),
                        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.9, 1.1))
                    ]),

            "val": None
        },
    "N-MNIST":
        {
            "train":T.Compose([
                        T.RandomAffine(degrees=10, translate=(0.1, 0.1), scale=(0.95, 1.05))
                    ]),

            "val": None
        },
}

train_transform = DATASET_TRANSFORM["CIFAR-10"]["train"]
val_transform = DATASET_TRANSFORM["CIFAR-10"]["val"]


def build_binary_augmentation(affine_degrees=0.0, affine_translate=0.0, affine_scale=0.0, erase_p=0.0):
    """Extra train-time augmentation for binary spike tensors, or None when all are off.

    RandomAffine resamples with nearest neighbour and fills with 0, and
    RandomErasing fills with 0, so the tensors stay binary.
    """
    transforms = []
    if affine_degrees or affine_translate or affine_scale:
        transforms.append(T.RandomAffine(
            degrees=affine_degrees,
            translate=(affine_translate, affine_translate) if affine_translate else None,
            scale=(1 - affine_scale, 1 + affine_scale) if affine_scale else None,
        ))
    if erase_p:
        transforms.append(T.RandomErasing(p=erase_p))
    return T.Compose(transforms) if transforms else None

__all__ = [
    "BinaryEmbedding",
    "ActiveSpatialCrop",
    "DATASET_TRANSFORM",
    "train_transform",
    "val_transform",
    "Denoise",
    "EventTransformCompose",
    "RandomFlipLR",
    "SpatialJitter",
    "TensorEventTransformCompose",
    "TensorRandomFlipLR",
    "TensorSpatialJitter",
    "build_binary_augmentation",
]
