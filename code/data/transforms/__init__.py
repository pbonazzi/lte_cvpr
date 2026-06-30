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
]
