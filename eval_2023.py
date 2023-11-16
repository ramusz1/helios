from argparse import ArgumentParser
import numpy as np
from pathlib import Path
import pandas as pd
import os
import matplotlib.pyplot as plt
import cv2
from src.util import load_rects, draw_rect, Scene, ImageLoader
from src.HyperTools import compile_results


parser = ArgumentParser(
    """
    View predicted bounding boxes and calculate prediction score.
    Example runs:
    > python eval_2023.py /scratch/datasets/hot_2023/training/HSI-VIS/automobile  --viz
    """)
parser.add_argument("input", help="Path to a directory containing input images and groundtruth_rect.txt")
parser.add_argument("--predictions", nargs="*", help="Path to the predictions file")
parser.add_argument("--models", nargs="*")
parser.add_argument("--viz", action="store_true", help="Display bounding boxes")
parser.add_argument("--step-by-step", action="store_true", help="Display the vizualization step by step, use [a,d] t switch frames")
parser.add_argument("--nostats", action="store_true")
parser.add_argument("--frames", help="Display custom dim reduction")

args = parser.parse_args()

predictions = args.predictions
models = args.models
if models is None:
    models = []
input_path = Path(args.input)
camera_type = input_path.parent.name
print(camera_type)
assert camera_type in ["HSI-VIS", "HSI-NIR", "HSI-RedNIR"]
scene = Scene(input_path.parent.parent, camera_type, input_path.name) 
y_true = scene.y_true
y_pred = []
if models is not None:
    for m in models:
        y_pred.append(load_rects(f"outputs/model_predictions/{m}/{input_path.parent.parent.name}/{camera_type}/{scene.name}.txt"))
if predictions is not None:
    for i,p in enumerate(predictions):
        y_pred.append(load_rects(p))
        models.append(os.path.basename(p))
img_shape = scene.hsi[0].shape[:2]
colors = [(255,0,0), (0,0,255), (255,255,0), (255,0,255), (0,255,255), (115,145,0), (145,0,115), (0, 115, 145)]

falsecolor = scene.falsecolor
falsecolor = [falsecolor, falsecolor]
falsecolor_names = ["falsecolor", "falsecolor"]
if args.frames:
    falsecolor[1] = ImageLoader(os.path.join(args.frames, scene.camera_type, scene.name ), "-FalseColor")
    falsecolor_names[1] = "custom falsecolor"


if args.viz:
    i = 0
    cv2.namedWindow('eval', cv2.WINDOW_KEEPRATIO)
    while i < len(falsecolor[0]):
        img = falsecolor[0][i]
        if i == 0:
            img = draw_rect(img, scene.init_bbox, (0, 255, 0))
        elif y_true is not None:
            img = draw_rect(img, y_true[i], (0, 255, 0))
        for j in range(len(y_pred)):
            img = draw_rect(img, y_pred[j][i], colors[j])

        font = cv2.FONT_HERSHEY_SIMPLEX
        th = 14 # text height
        cv2.putText(img, f"{i:04d}", (0, th), font, 0.5, (50,200,50), 1, cv2.LINE_AA)
        for j in range(len(y_pred)):
            cv2.putText(img, models[j], (0, th * (j + 2)), font, 0.5, colors[j], 1, cv2.LINE_AA)
            
        cv2.putText(img, falsecolor_names[0], (0, img.shape[0]-1), font, 0.5, (0,255,0), 1, cv2.LINE_AA)

        cv2.imshow("eval", img)
        key = (cv2.waitKey(20) & 0xFF)
        if key == ord('q'):
            break
        elif key == ord('f'):
            falsecolor = falsecolor[::-1]
            falsecolor_names = falsecolor_names[::-1]
        elif key == ord('a'):
            i = max(0, i - 1)
        elif key == ord('d'):
            i = i + 1
        elif not args.step_by_step:
            i = i + 1
 
if len(models) > 0 and args.nostats is False:

    fig, axs = plt.subplots(1, 2, figsize=(15,5), num="eval")
    auc = np.zeros(len(y_pred))
    dp_20 = np.zeros(len(y_pred))

    for j in range(len(y_pred)):
        precision, success, average_center_location_error, auc[j], dp_20[j] = compile_results(y_true, y_pred[j])
        c = np.array(colors[j]) / 255.0
        axs[0].plot(precision[0], precision[1], label=models[j], c=c)
        axs[1].plot(success[0], success[1], label=models[j], c=c)

    axs[0].set_xlabel("Threshold")
    axs[0].set_ylabel("% of predictions\nwithin distance\nthreshold", rotation=0, ha='right')
    axs[0].set_ylim(-0.05,1.05)
    axs[0].set_title("Precision plot")
    axs[0].legend()

    axs[1].set_xlabel("IOU threshold")
    axs[1].set_ylabel("% of predictions\nwith IOU greater than\n IOU threshold", rotation=0, ha='right')
    axs[1].set_ylim(-0.05,1.05)
    axs[1].set_title("Success plot")
    axs[1].set_aspect("equal")
    axs[1].legend()

    df = pd.DataFrame({"model":models, "auc":auc, "dp_20":dp_20}) 
    fig.suptitle(df.to_string(index=False))
    fig.tight_layout()

    plt.show()
