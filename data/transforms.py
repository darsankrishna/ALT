# data/transforms.py
import torchvision.transforms as T

CLIP_MEAN = [0.48145466, 0.4578275,  0.40821073]
CLIP_STD  = [0.26862954, 0.26130258, 0.27577711]

def get_train_transforms(size: int = 224) -> T.Compose:
    """Training transforms for CLIP-normalized frames `(3, H, W)`."""
    return T.Compose([
        T.RandomResizedCrop(size, scale=(0.5, 1.0)),
        T.RandomHorizontalFlip(),
        T.ColorJitter(brightness=0.4, contrast=0.4, saturation=0.4, hue=0.1),
        T.RandomGrayscale(p=0.2),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])

def get_val_transforms(size: int = 224) -> T.Compose:
    """Validation transforms for CLIP-normalized frames `(3, H, W)`."""
    return T.Compose([
        T.Resize(int(size * 256 / 224)),
        T.CenterCrop(size),
        T.Normalize(mean=CLIP_MEAN, std=CLIP_STD),
    ])
