# Breast Cancer Molecular Subtypes Classifcation

 This repository contains the implementation of our DL method for Breast Mass Segmentation, feature extraction methods and advanced machine-learning techniques for classifying breast cancer subtypes.

## Key Highlights

- **Custom U-Net-Based Autoencoder:** Developed a custom U-Net-based autoencoder with boundary-guided skip connections. This approach enhances tumor boundary detection, significantly reducing segmentation inaccuracies from the boundaries.
- **Feature Extraction:** Extracted novel combinations of morphological and radiomics features from accurately segmented tumor regions, facilitating detailed assessments of tumor aggressiveness and molecular subtype.
- **Comprehensive Analyses:** Performed comprehensive analyses correlating tumor shape, margins, and radiomics to breast cancer molecular subtypes.
- **Sophisticated ML Framework:** Developed a sophisticated machine learning framework for highly accurate classification of breast cancer molecular subtypes.

## Repository Structure

- `seg_model.py` - Contains the segmentation model with all required modules.
- `training.py` - Used to run the experiments for training the models.
- `morphological_feature_extraction.py` - Functions to extract morphological features from the tumor region.
- `texture_features_extraction.py` - Functions to extract different texture features.
- `ml_pipeline.py` - Utilizes the extracted features to classify tumor shape, margins, and molecular subtypes.

## Setup and Installation

Ensure you have the following prerequisites installed:
- Python 3.8
- TensorFlow 2.2

## Contributing
We welcome contributions to this repository. If you have suggestions or improvements, please contact.

## Citation
If you use the code or methodologies from this repository in your research, please cite our research.

### Dataset Information

All experiments conducted as part of this research utilize our private in-house dataset. This dataset is managed under strict compliance with relevant data protection laws to ensure privacy and security. Access to this dataset is limited to researchers directly involved in the project and is not available publicly to ensure the confidentiality and integrity of the data.


### Protection of Personal Data
Our data handling practices adhere to the highest standards of personal data protection, in compliance with Article 7.1 of the Organic Law 15/1999. We ensure that all personal information is securely managed and accessed only by authorized personnel involved in the research.

$$License
See LICENSE for more information.
