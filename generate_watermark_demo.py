from utils.preprocess_text import preprocess_txt
from watermark_model import watermark_model as repeat_no_context_model_bert


def watermark_embed_demo(raw):
    watermarked_text = model.embed(raw)
    return watermarked_text


def repeat_watermark_detect(raw):
    is_watermark, p_value, n, ones, z_value = model.watermark_detector(raw)
    confidence = (1 - p_value) * 100
    return z_value, confidence, is_watermark


if __name__ == "__main__":

    text = ''
    model = repeat_no_context_model_bert()

    original_text = preprocess_txt(text)
    watermark_text = model.embed(original_text)
    z_value, confidence, is_watermark = repeat_watermark_detect(watermark_text)

    print(f'===\nWatermark Text: {watermark_text}\nz_value: {z_value}\nconfidence: {confidence}\nis_watermark: {is_watermark}\n===')

