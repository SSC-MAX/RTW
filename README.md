# RTW
Robust Text Watermarking for Black-box Large Language Model

# Environment
Run `pip install -r requirements.txt` to download all required dependencies.  

# Training SVMM
Run `utils/train_watermark_model.py` to train the model. All trainning data can be found in `train_data`.
`models/transform_model.pth` provides a trained model which was used in paper.

# Watermark Embedding and Extraction
Run `generate_watermark.py` to embed watermark.  
Run `evaluation.py` to calcualte F1 score and AUC.