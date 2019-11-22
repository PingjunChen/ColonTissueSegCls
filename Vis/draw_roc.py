# -*- coding: utf-8 -*-

import os, sys
import json
import numpy as np
from sklearn.metrics import roc_curve, auc
import matplotlib.pyplot as plt
from mpl_toolkits.axes_grid1.inset_locator import zoomed_inset_axes
from mpl_toolkits.axes_grid1.inset_locator import mark_inset
from matplotlib.backends.backend_pdf import PdfPages

def load_gt_prob(json_path):
    with open(json_path) as fp:
        data_dict = json.load(fp)
        gt = np.asarray(data_dict["gts"])[:, 1]
        prob = np.asarray(data_dict["preds"])[:, 1]
    return gt, prob



def draw_roc(pred_dict):
    #Plot all roc curve
    fig, axes = plt.subplots(nrows=1, ncols=2, figsize=(12, 5))
    with PdfPages('colon_roc.pdf') as pdf:
        for i, (k, v) in enumerate(pred_dict.items()):
            gt_list, prob_list = v["gts"], v["probs"]
            split_num = len(gt_list)
            axes[i].plot([0, 1], [0, 1], 'k--')
            axins = zoomed_inset_axes(axes[i], 2.5, loc="center right")
            auc_list = []
            for ind in range(split_num):
                fpr_rate, tpr_rate, thresholds = roc_curve(gt_list[ind], prob_list[ind])
                auc_list.append(auc(fpr_rate, tpr_rate))
                axes[i].plot(fpr_rate, tpr_rate)
                axins.plot(fpr_rate, tpr_rate)
            axes[i].text(0.6, 0.05, "mAUC: {:.3f}".format(np.mean(auc_list)), fontsize=12)
            axes[i].set_xlabel('False positive rate')
            axes[i].set_ylabel('True positive rate')
            axes[i].set_title(k)
            x1, x2, y1, y2 = 0.0, 0.36, 0.88, 1.0 # specify the limits
            axins.set_xlim(x1, x2) # apply the x-limits
            axins.set_ylim(y1, y2) # apply the y-limits
            mark_inset(axes[i], axins, loc1=3, loc2=1, fc="none", ec="0.5")
        # plt.show()
        plt.tight_layout()
        pdf.savefig()


if __name__ == "__main__":
    splits = ["00", "01", "02", "03", "04"]
    fusion_methods = ["pooling", "selfatt"]

    pred_dict = {}
    for method in fusion_methods:
        gt_list, prob_list = [], []
        for split in splits:
            json_path = os.path.join("./data", method + split + "_pred_gt.json")
            cur_gt, cur_prob = load_gt_prob(json_path)
            gt_list.append(cur_gt)
            prob_list.append(cur_prob)
        pred_dict[method] = {"gts": gt_list, "probs": prob_list}
    draw_roc(pred_dict)
