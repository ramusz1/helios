import os
import argparse
import numpy as np

from src.dim_reduction import DimReduceV4, XGBBandSelectorV6, ImadjustV5, BilateralFilter
from src.dim_reduction_util import hsi_to_rgb
from src.tracker import OSTrackWrapper
from src.forward_backward import backward_forward
from src.util import save_rects, ImageLoader



# argument parsing

parser = argparse.ArgumentParser(
        """
        Run HELIOS tracker on a hyperspectral sequence.
        Example usage:
        > python run.py /scratch/datasets/hot_2023/validation/HSI-VIS/coin  --init_bbox 96 83 16 16 --camera_type HSI-VIS --scene_name coin
        """)
parser.add_argument("hsi", help="diretory with hyperspectral images")
parser.add_argument("--init_bbox", nargs=4, type=int, help="initial bounding box (x,y,w,h)", required=True)
parser.add_argument("--camera_type", choices=["HSI-VIS", "HSI-NIR", "HSI-RedNIR"], required=True)
parser.add_argument("--falsecolor", help="directory with falsecolor images")
parser.add_argument("--scene_name", default="test_scene", help="scene name for saving predictions and debug")


args = parser.parse_args()

# instatntiate tracker and dim reduction

dim_reduction = DimReduceV4(
    XGBBandSelectorV6(alpha=0.005),
    ImadjustV5(0.1, (0.005, 0.995), (0.005, 0.995), 0.5, debug=False),
    BilateralFilter()
)

ostrack = OSTrackWrapper(
    use_hann=True,
    candidate_bbox_threshold=0.7,
    nms_iou_thresh=0.7,
    max_candidates=3
)

# prepare run arguments

output_dir = f"outputs/model_predictions/helios/"
pred_path = f"{output_dir}/{args.scene_name}.txt"

hsi_loader = ImageLoader(args.hsi, args.camera_type)
if args.falsecolor:
    falsecolor_loader = ImageLoader(args.falsecolor, args.camera_type + "-FalseColor")
else:

    class FalsecolorFactory:
        def __init__(self, hsi_loader):

            self.hsi_loader = hsi_loader

        def __len__(self):
            return len(self.hsi_loader)

        def __getitem__(self, i):
            # f = hsi_to_rgb(self.hsi_loader[i])
            # cv2.imshow("dbg", f)
            # cv2.waitKey(0)
            # return f
            return hsi_to_rgb(self.hsi_loader[i])
        
    falsecolor_loader = FalsecolorFactory(hsi_loader)

frames = hsi_loader , falsecolor_loader 
bbox_init = np.array(args.init_bbox)
forward = backward_forward(ostrack, dim_reduction, frames, bbox_init, args.camera_type, args.scene_name, debug=False, maxiter=3)

os.makedirs(output_dir, exist_ok=True)
save_rects(pred_path, forward)

