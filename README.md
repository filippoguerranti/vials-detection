# Vial detection

## Goal

Implemention of an end-to-end model for the classification pharmaceutical vials during different reconciliation steps of a production chain.

## Project workflow

1. Creation of an ad-hoc dataset, called **VIALS** dataset
2. Implementation of a Supervised Convolutional Autoencoder (SAE)
    * `SAE32`: handles 32x32 images
    * `SAE90`: handles 90x90 images 
3. Implementation of a baseline Convolutional Neural Network (CNN)
    * `CNN32`: handles 32x32 images
    * `CNN90`: handles 90x90 images 
4. Data augmentation and batch balance to handle class imbalance and similarity among elements of same class
5. Comparison of the models

## Info

*Author*: Filippo Guerranti ([email](mailto:filippo.guerranti@student.unisi.it))  
*Professor*: Alessandro Mecocci ([email](mailto:mecocci@unisi.it))  
*Exam*: Advanced Digital Image Processing  
*In collaboration with*: Pharma Integration ([website](http://www.pharma-integration.it))
