import argparse

from scipy import interpolate
from sklearn.metrics import roc_auc_score, roc_curve, precision_recall_curve
import numpy as np

import json

def tpr_at_fpr(fpr, tpr, fpr_target):
    fpr_tpr_interpolation = interpolate.interp1d(fpr, tpr, kind='linear')
    return fpr_tpr_interpolation(fpr_target)

def f1_at_fpr(y_true, y_scores, fpr_target):
    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    # Finding the threshold for our target FPR
    threshold = thresholds[next(i for i in range(len(fpr)) if fpr[i] > fpr_target) - 1]
    precision, recall, thresholds_pr = precision_recall_curve(y_true, y_scores)

    # Interpolating to find precision and recall at the target threshold
    precision_interp = interpolate.interp1d(thresholds_pr, precision[:-1], fill_value="extrapolate")
    recall_interp = interpolate.interp1d(thresholds_pr, recall[:-1], fill_value="extrapolate")
    precision_at_threshold = precision_interp(threshold)
    recall_at_threshold = recall_interp(threshold)
    # Calculate F1 score
    f1 = 2 * (precision_at_threshold * recall_at_threshold) / (precision_at_threshold + recall_at_threshold)

    return f1

def cal_f1(detect_data, water_value, ori_value, f1_file):
    # non-watermark
    hm_zscore = [x[ori_value] if not np.isnan(x[ori_value]) else 0 for x in detect_data]
    hm_true = [0 for x in detect_data]

    # watermarked
    wm_zscore = [x[water_value] if not np.isnan(x[water_value]) else 0 for x in detect_data]
    wm_true = [1 for x in detect_data]

    y_true = hm_true + wm_true
    y_scores = hm_zscore + wm_zscore

    auc = roc_auc_score(y_true, y_scores)

    fpr, tpr, thresholds = roc_curve(y_true, y_scores)

    print(f""" {water_value}

        AUC: {auc:.3f}

    TPR@FPR=0.05: {tpr_at_fpr(fpr, tpr, 0.05):.3f}
    TPR@FPR=0.01: {tpr_at_fpr(fpr, tpr, 0.01):.3f}

    F1@FPR=0.05: {f1_at_fpr(y_true, y_scores, 0.05):.3f}
    F1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}
    """
          )

    f1_data = f'F1@FPR=0.05: {f1_at_fpr(y_true, y_scores, 0.05):.3f}\nF1@FPR=0.01: {f1_at_fpr(y_true, y_scores, 0.01):.3f}\nauc:{auc:.3f}'
    with open(f1_file, 'w', encoding='utf-8') as file:
        file.write(f1_data)
    return fpr, tpr, thresholds


def main(args, data_size):
    import json
    with open(args.zscore, 'r', encoding="utf-8") as file:
        detect_data = json.load(file)
    if data_size>len(detect_data):
        print("Amount of data is too large")
        return 1
    detect_data = detect_data[0:data_size]

    print(f'data_size => {args.data_size}')
    
    fpr, tpr, thresholds = cal_f1(detect_data, f'{args.compare_type}-fast-z-score', 'ori-fast-z-score', args.f1_file)


if __name__ == "__main__":
    
    file_name = ''   # result ouput by RTW
    f1_file = ''     # result of F1 and AUC

    data_size = len(json.load(open(file_name)))
    
    parser = argparse.ArgumentParser(description='Generate with watermarking')
    # parser.add_argument("--hm_zscore", type=str, required=True, help="Human zscore file")
    # parser.add_argument("--wm_zscore", type=str, required=True, help="Watermark zscore file")
    parser.add_argument("--zscore", type=str, required=False, help="input file",default=file_name)
    parser.add_argument("--roc_curve", type=str, default="1", help="ROC curve file")
    parser.add_argument('--data_size', type=int, required=False, help="A output json file of list of strings.",
                        default=data_size)
    parser.add_argument('--compare_type', type=str, default='water')
    parser.add_argument('--f1_file', type=str, default=f1_file)

    args = parser.parse_args()
    main(args,args.data_size)