# RTW
Robust Text Watermarking for Black-box Large Language Model

# Training SVMM
Run `utils/train_watermark_model.py` to train the model. `train_data` contains all trainning data.
`models/transform_model.pth` provides a trained model which was used in paper.

# Watermark Embedding and Extraction
Run `generate_watermark.py` to embed watermark.  
Run `evaluation.py` to extract and calcualte F1 score and AUC.