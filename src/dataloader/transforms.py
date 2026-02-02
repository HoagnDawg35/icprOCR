import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_train_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),

        # --- Geometric Distortions ---
        A.OneOf([
            A.Affine(
                scale=(0.9, 1.1),
                translate_percent=(0.05, 0.05),
                rotate=(-10, 10),
                shear=(-5, 5),
            ),
            A.Perspective(scale=(0.02, 0.08)),
            A.GridDistortion(num_steps=5, distort_limit=0.3),
        ], p=0.7),

        # --- Quality & Detail ---
        A.OneOf([
            A.CLAHE(clip_limit=2.0, tile_grid_size=(8, 8)),
            A.Sharpen(alpha=(0.2, 0.5), lightness=(0.5, 1.0)),
            A.Emboss(alpha=(0.2, 0.5), strength=(0.2, 0.7)),
            A.RandomGamma(gamma_limit=(80, 120)),
        ], p=0.3),

        # --- Color ---
        A.OneOf([
            A.RandomBrightnessContrast(
                brightness_limit=0.2,
                contrast_limit=0.2
            ),
            A.HueSaturationValue(
                hue_shift_limit=10,
                sat_shift_limit=20,
                val_shift_limit=20
            ),
        ], p=0.5),

        # --- Noise & Artifacts ---
        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.ImageCompression(
                quality_lower=60,
                quality_upper=100
            ),
            A.CoarseDropout(
                max_holes=2,
                max_height=4,
                max_width=4,
                fill_value=(128, 128, 128)
            ),
        ], p=0.3),

        A.ChannelShuffle(p=0.2),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_light_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2(),
    ])


def get_degradation_transforms() -> A.Compose:
    return A.Compose([
        A.OneOf([
            A.GaussianBlur(blur_limit=(3, 5)),
            A.MotionBlur(blur_limit=(3, 5)),
        ], p=0.7),

        A.OneOf([
            A.GaussNoise(var_limit=(10.0, 50.0)),
            A.MultiplicativeNoise(multiplier=(0.9, 1.1)),
        ], p=0.7),

        A.ImageCompression(
            quality_lower=20,
            quality_upper=50,
            p=0.5
        ),

        A.Downscale(
            scale_min=0.3,
            scale_max=0.5,
            p=0.5
        ),
    ])


def get_val_transforms(img_height: int = 32, img_width: int = 128) -> A.Compose:
    return A.Compose([
        A.Resize(height=img_height, width=img_width),
        A.Normalize(mean=(0.5, 0.5, 0.5),
                    std=(0.5, 0.5, 0.5)),
        ToTensorV2()
    ])
