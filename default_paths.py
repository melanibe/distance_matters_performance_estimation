from pathlib import Path

ROOT = Path(__file__).parent
DATA_CIFARC = Path("/data/failure_detection/data/CIFAR-10-C")
DATA_WILDS_ROOT = "/data/wilds" if Path("/data/wilds").exists() else "/vol/biodata/data/wilds"
DATA_MEDMNIST = ROOT / "data"
DATA_IMAGENET = Path("/data/ILSVRC2012") if Path("/data/ILSVRC2012").exists() else Path("/vol/biodata/data/ILSVRC2012")
DATA_IMAGENET_SKETCH = Path("/vol/biodata/data/imagenet-sketch/sketch")
DATA_MNIST = Path("/vol/biodata/data/mnist-variants")
BREEDS_INFO_DIR = ROOT / "data_handling" / "imagenet_class_hierarchy" / "modified"
DATA_PACS = ROOT / "data" / "pacs"
DATA_IMAGENET_V2 = Path("/vol/biodata/data/imagenetv2")
DATA_IMAGENET_A = Path("/data/imagenet-a")
DATA_VAL_IMAGENET200A = Path("data/imagenet_val_for_imagenet_a")
