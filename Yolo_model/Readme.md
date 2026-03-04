# Trained YOLO Model (Ultralytics)

This repository provides a trained Ultralytics YOLO detection model.

## Requirements

- Python 3.9 or newer recommended
- Ultralytics package

Install Ultralytics:
```bash
pip install ultralytics
```
### Prepare the dataset YAML 
Make sure dataset.yaml exists in the repository root and points to your dataset.
```bash
Example dataset.yaml:
path: /absolute/or/relative/path/to/dataset
train: images/train
val: images/val
test: images/test

names:
  0: object
```
Notes:
path can be absolute or relative.

### Validation script

```bash
from ultralytics import YOLO
import os

# Path
script_dir = os.path.dirname(os.path.abspath(__file__))
data_path = os.path.join(script_dir, "my_dataset.yaml")

# Load my own model
model = YOLO("runs/detect/train/weights/best.pt")

# To resolve a RuntimeError when launching in Windows, we add this before
if __name__ == "__main__":
    test_results = model.val(
        data=data_path,
        batch=16,
        conf=0.1,
        plots=True,
        split="test",
        single_cls=True,
    )
```

#### Outputs

When plots=True, Ultralytics typically saves:
 - confusion matrix
 - precision recall curve
 - F1 curve=
 - validation metrics summary
> The exact output location depends on your Ultralytics version and run settings.
train, val, and test are paths relative to path.

If you use single_cls=True, evaluation treats all labels as one class. You can still keep names defined.
