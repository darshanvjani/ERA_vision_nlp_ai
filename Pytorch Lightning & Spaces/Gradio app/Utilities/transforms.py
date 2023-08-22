import albumentations as A
from albumentations.pytorch import ToTensorV2

# Define the transforms (only test)

test_transforms = A.Compose([

    A.Normalize((0.4914, 0.4822, 0.4465), (0.247, 0.243, 0.261)),
    ToTensorV2()

])