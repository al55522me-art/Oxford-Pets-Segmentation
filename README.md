**Multi-Class Semantic Segmentation of Animals using U-Net**

This project implements semantic segmentation of animals in images from the [Oxford-IIIT Pets Dataset](https://www.robots.ox.ac.uk/~vgg/data/pets/).  
The **U-Net** model is trained to separate animals (cats and dogs) from the background using annotated masks.
## Prediction example

![Overlay Sample](outputs/prediction.png)

| Overlay | Project Description |
|-----|------------|
| ![Segmentation Demo](outputs/segmentation_demo1.gif) | This project implements **multi-class semantic segmentation** of animals from the Oxford-IIIT Pets dataset.<br><br>**Project Goals:**<br><br>- Learn and apply the U-Net architecture for multi-class segmentation.<br>- Gain hands-on experience working with images and masks.<br>- Visualize model predictions with overlays 
 
**Metrics**
- Validation Dice Score: ~ 0.89

**Model Download**
- https://drive.google.com/file/d/1xhuBg5zvdtYcTmorcI9HSFXmFLkBvpTC/view?usp=share_link

## Visualization Demo

```python
from src.dataset import OxfordPetsMultiClassDataset
from src.model import UNetMultiClass
from src.visualize import visualize_predictions_multiclass
import torch

# Setup device
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load the model
model = UNetMultiClass(n_channels=3, n_classes=3).to(device)
checkpoint = torch.load("outputs/best_model_multiclass.pth", map_location=device)
model.load_state_dict(checkpoint["model_state_dict"])

# Load the dataset
dataset = OxfordPetsMultiClassDataset(root="oxford_pets")

# Visualize 5 sample predictions
visualize_predictions_multiclass(model, dataset, device, num_samples=5)
