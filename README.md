# Attention-guided-Adaptive-Fusion
Abstract:This paper presents a framework for adaptive multimodal feature fusion that employs attention-based feature selection mechanisms to enhance the classification of histopathology images. The framework integrates medical images and clinical texts through three core modules: the Unified Feature Processing Module (UFPM) for standardized feature preprocessing, the Cross-Modal Attention Module (CMAM) for facilitating interactions between image and text features, and the Selective Feature Alignment Module (SFAM) for aligning features across different modalities. Experimental results on the Quilt-BCGG and Quilt-Derm4 datasets demonstrate the framework's good classification performance, which improves the accuracy and efficiency of histopathology diagnosis through optimized feature selection and alignment. 
# Usage
  1.Train single-modal models:

      # Train image classification model
      python mul_img_classify.py
      # Train text classification model
      python mul_txt_classify.py
  2.Save pre-trained models:

      # Save image model
      torch.save(model.state_dict(), 'path/to/image_model.pth')
      # Save text model
      torch.save(model.state_dict(), 'path/to/text_model.pth')
  3.Train multimodal fusion model:

      # Set pre-trained model paths
      text_model_path = 'path/to/text_model.pth'
      image_model_path = 'path/to/image_model.pth'
      # Train multimodal model
      python multimodal_classify.py
