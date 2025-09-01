# Breast Cancer Molecular Subtypes Classification

This repository implements a deep learning based framework for breast mass segmentation and molecular subtype classification. It uses a boundary aware autoencoder with guided skip connections to delineate tumor margins, extracts morphological and radiomic features, analyzes correlations between shape, margins, and radiomic patterns, and applies a high accuracy machine learning pipeline such as CatBoost for noninvasive subtype prediction.

Model weights are available [here](https://zenodo.org/records/17014860).

## Repository Structure

* `seg_model.py`: segmentation model with all required modules  
* `seg_pred.py`: run segmentation tests and save predicted masks  
* `morphological_feature_extraction.py`: compute morphological features from tumor masks  
* `texture_features_extraction.py`: compute texture and radiomic features  
* `Subtype_Classifcation.py`: load the saved CatBoost model and predict molecular subtypes  
* `Saved_Model/`: saved classifier models trained on extracted features

---
## Setup and Installation
**Prerequisites**

* Python 3.8  
* TensorFlow 2.14.0  
* Recommended Python packages: numpy, scikit learn, opencv python, catboost, matplotlib

**Install**
Ensure you have the following prerequisites installed:
- Python 3.8
- TensorFlow 2.14.0
- pip install tensorflow==2.14.0 numpy scikit-learn opencv-python catboost matplotlib

## Contributing
We welcome contributions to this repository. If you have suggestions or improvements, please contact.

## Citation
If you use the code or methodologies from this repository in your research, please cite our research.

### Dataset Information

All experiments conducted as part of this research utilize our private in-house dataset. This dataset is managed under strict compliance with relevant data protection laws to ensure privacy and security. Access to this dataset is limited to researchers directly involved in the project and is not available publicly to ensure the confidentiality and integrity of the data.


### Protection of Personal Data
Our data handling practices adhere to the highest standards of personal data protection, in compliance with Article 7.1 of the Organic Law 15/1999. We ensure that all personal information is securely managed and accessed only by authorized personnel involved in the research.

