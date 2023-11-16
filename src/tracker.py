import torch
import importlib
from torchvision.ops import nms
import abc
import os
from tqdm import trange

from pathlib import Path
import sys

sys.path.append(str(Path(__file__).parent.parent.joinpath("ext/OSTrack")))

from ext.OSTrack.lib.vis.visdom_cus import Visdom
from lib.models.ostrack import build_ostrack
from lib.test.tracker.basetracker import BaseTracker
from lib.test.utils.hann import hann2d
from lib.train.data.processing_utils import sample_target
from lib.test.tracker.data_utils import Preprocessor
from lib.utils.box_ops import clip_box
from lib.utils.ce_utils import generate_mask_cond

from .util import bbox_out_of_bounds, bbox_completely_out_of_bounds, save_rects



class BaseHOTTracker(metaclass=abc.ABCMeta):


    @abc.abstractmethod
    def initialize(self, frame, init_bbox, scene_name):
        pass

    @abc.abstractmethod
    def pred(self, frame):
        return None


class BaseOSTracker(BaseHOTTracker):
    
    def __init__(self, debug_log=False, debug_images=False, use_visdom=False):
        self.debug_log = debug_log
        self.debug_images = debug_images
        self.use_visdom = use_visdom
        if self.use_visdom:
            self._init_visdom(None)
        else:
            self.visdom = None
        
    def pred(self, frame):
        out = self._pred(frame)
        if self.use_visdom:
            while self.pause_mode:
                sleep(0.001) # sleep 1 ms
                if self.step:
                    self.step = False
                    break
        return out

    def _pred(self, frame):
        raise NotImplemented()
    
    
    def _init_visdom(self, visdom_info):
        visdom_info = {} if visdom_info is None else visdom_info
        self.pause_mode = False
        self.step = False
        self.next_seq = False
        self.visdom = Visdom(1, {'handler': self._visdom_ui_handler, 'win_id': 'Tracking'},
                             visdom_info=visdom_info)
        help_text = 'You can pause/unpause the tracker by pressing ''space'' with the ''Tracking'' window ' \
                    'selected. During paused mode, you can track for one frame by pressing the right arrow key.' \
                    'To enable/disable plotting of a data block, tick/untick the corresponding entry in ' \
                    'block list.'
#         self.visdom.register(help_text, 'text', 1, 'Help')

    def _visdom_ui_handler(self, data):
        if data['event_type'] == 'KeyPress':
            if data['key'] == ' ':
                self.pause_mode = not self.pause_mode

            elif data['key'] == 'ArrowRight' and self.pause_mode:
                self.step = True 

            elif data['key'] == 'n':
                self.next_seq = True
                
    def log(self, *args, **kwargs):
        if self.debug_log:
            print(*args, **kwargs)
   

class OSTrackWrapper(BaseTracker):
    
    def __init__(self,
            use_hann=True, candidate_bbox_threshold=0.8,
            nms_iou_thresh=0.7, max_candidates=3, preprocessor=None,
            debug_images=False, visdom=None):
        params = self._get_network_params() 
        super(OSTrackWrapper, self).__init__(params)
        network = build_ostrack(params.cfg, training=False)
        network.load_state_dict(torch.load(self.params.checkpoint, map_location='cpu')['net'], strict=True)
        self.cfg = params.cfg
        self.network = network.cuda()
        self.network.eval()
        if preprocessor is None:
            preprocessor = Preprocessor()
        self.preprocessor = preprocessor
        self.candidate_bbox_threshold = candidate_bbox_threshold
        self.nms_iou_thresh = nms_iou_thresh
        self.max_candidates = max_candidates
        assert params.cfg.MODEL.HEAD.TYPE == "CENTER"

        self.feat_sz = self.cfg.TEST.SEARCH_SIZE // self.cfg.MODEL.BACKBONE.STRIDE
        # motion constrain
        self.output_window = hann2d(torch.tensor([self.feat_sz, self.feat_sz]).long(), centered=True).cuda()
        self.use_hann = use_hann
        
        # for debug
        self.visdom = visdom
        self.debug_images = debug_images
        self.use_visdom = self.visdom is not None

        self.z_dict1 = {}
        
    def _get_network_params(self):
        """Get parameters."""
        name = "ostrack"
        parameter_name = "vitb_384_mae_ce_32x4_ep300"
        param_module = importlib.import_module('lib.test.parameter.{}'.format(name))
        params = param_module.parameters(parameter_name)
        params.tracker_name = name
        params.param_name = parameter_name
        return params
        
    def initialize(self, image, init_bbox):
        H, W, _ = image.shape
        # forward the template once
        if bbox_out_of_bounds(init_bbox, H, W):
            print(init_bbox, H, W)
            raise ValueError("init bbox out of bounds")
        z_patch_arr, resize_factor, z_amask_arr = sample_target(image, init_bbox, self.params.template_factor,
                                                    output_sz=self.params.template_size)
        self.z_patch_arr = z_patch_arr
        template = self.preprocessor.process(z_patch_arr, z_amask_arr)
        with torch.no_grad():
            self.z_dict1 = template

        self.box_mask_z = None
        if self.cfg.MODEL.BACKBONE.CE_LOC:
            template_bbox = self.transform_bbox_to_crop(init_bbox, resize_factor,
                                                        template.tensors.device).squeeze(1)
            self.box_mask_z = generate_mask_cond(self.cfg, 1, template.tensors.device, template_bbox)
    
    """
    score_map : a visual prediction map of the current position of the target, centered around the previous position
    """
    def extract_bounding_boxes(self, score_map_ctr, size_map, offset_map):
        assert score_map_ctr.shape[0] == 1 and score_map_ctr.shape[1] == 1
        feat_sz = score_map_ctr.shape[-1]
        score_map_ctr_flat = score_map_ctr.flatten(1).clone()
        # calculate area map and set score to 0 if area is 0
        size_map_flat = size_map.flatten(2)
        area_map_flat = torch.clamp(size_map_flat[:,0], 0) * torch.clamp(size_map_flat[:,1], 0)
        score_map_ctr_flat[area_map_flat <= 0] = 0
        # currently uses the implementation from CenterPredictor in head.py
        max_score, idx = torch.max(score_map_ctr_flat, dim=1, keepdim=True)
        idx = torch.nonzero(score_map_ctr_flat >= max_score * self.candidate_bbox_threshold, as_tuple=True)[1]
        idx = idx.unsqueeze(0)
        idx_y = idx // self.feat_sz
        idx_x = idx % self.feat_sz
        
        scores = score_map_ctr_flat.gather(dim=1, index=idx)
        idx = idx.unsqueeze(1).expand(-1, 2, -1)
        sizes = size_map_flat.gather(dim=2, index=idx)
        offsets = offset_map.flatten(2).gather(dim=2, index=idx)

        bboxes = torch.cat([(idx_x.to(torch.float) + offsets[:, :1]) / self.feat_sz,
              (idx_y.to(torch.float) + offsets[:, 1:]) / self.feat_sz,
              sizes], dim=1)
        bboxes = torch.swapaxes(bboxes, 1, 2)
        bboxes = bboxes.view(-1, 4)
        scores = scores.view(-1)
        
        # non maximum suppression
        # expects x1,y1,x2,y2 format
        bboxes[:,2:] += bboxes[:,:2]
        idx = nms(bboxes, scores, iou_threshold=self.nms_iou_thresh)
        bboxes = bboxes[idx]
        scores = scores[idx].view(-1, 1)
        bboxes[:,2:] -= bboxes[:,:2] # back to x,y,w,h format
        # nms does the sorting
#         scores, idx = torch.sort(scores, descending=True, dim=0)
#         bboxes = bboxes.gather(dim=0, index=idx.expand(-1,4))
        if self.max_candidates is not None and scores.shape[0] > self.max_candidates:
            scores = scores[:self.max_candidates]
            bboxes = bboxes[:self.max_candidates]
        
        return bboxes, scores
    
    def track(self, image, previous_bbox, frame_id):
        H, W, _ = image.shape
        if bbox_out_of_bounds(previous_bbox, H, W):
            print(previous_bbox, H, W)
            raise ValueError("previous bbox out of bounds")
        x_patch_arr, resize_factor, x_amask_arr = sample_target(image, previous_bbox, self.params.search_factor,
                                                                output_sz=self.params.search_size)  # (x1, y1, w, h)
        search = self.preprocessor.process(x_patch_arr, x_amask_arr)

        with torch.no_grad():
            x_dict = search
            # merge the template and the search
            # run the transformer
            out_dict = self.network.forward(
                template=self.z_dict1.tensors, search=x_dict.tensors, ce_template_mask=self.box_mask_z)

        # add hann windows
        pred_score_map = out_dict['score_map']
        if self.use_hann:
            response = self.output_window * pred_score_map
        else:
            response = pred_score_map
            
        pred_boxes, scores = self.extract_bounding_boxes(
            response, out_dict['size_map'],
            out_dict['offset_map'])
        pred_boxes *= self.params.search_size / resize_factor
        
        # get the final box result
        offset = self.response_offset(resize_factor, previous_bbox)
        scale = 1.0 / (self.params.search_size / resize_factor)
        all_boxes = self.map_box_back_batch(pred_boxes, resize_factor, previous_bbox)
    
        if self.use_visdom:

            self.visdom.register(torch.from_numpy(x_patch_arr).permute(2, 0, 1), 'image', 1, 'search_region')
            self.visdom.register(torch.from_numpy(self.z_patch_arr).permute(2, 0, 1), 'image', 1, 'template')
            self.visdom.register(pred_score_map.view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map')
            self.visdom.register((pred_score_map * self.output_window).view(self.feat_sz, self.feat_sz), 'heatmap', 1, 'score_map_hann')

        return all_boxes, scores, {"response": response, "offset": offset, "scale": scale}
    
    def response_offset(self, resize_factor: float, state):
        cx_prev, cy_prev = state[0] + 0.5 * state[2], state[1] + 0.5 * state[3]
        half_side = 0.5 * self.params.search_size / resize_factor
        x_offset = cx_prev - half_side
        y_offset = cy_prev - half_side
        return x_offset, y_offset
    
    def map_box_back_batch(self, pred_box: torch.Tensor, resize_factor: float, state):
        x_offset, y_offset = self.response_offset(resize_factor, state)
        cx, cy, w, h = pred_box.unbind(-1) # (N,4) --> (N,)
        cx_real = cx + x_offset
        cy_real = cy + y_offset
        return torch.stack([cx_real - 0.5 * w, cy_real - 0.5 * h, w, h], dim=-1)




def run_on_dataset(tracker: BaseHOTTracker, model_name, dataset, dataset_type, scenes, falsecolor=True, skip_existing=False):
    for name in scenes:
        scene = dataset.get_scene(name)
        
        output_dir = f"../outputs/model_predictions/{model_name}/{dataset_type}/{dataset.camera_type}"
        pred_path = f"{output_dir}/{scene.name}.txt"
        if skip_existing and os.path.exists(pred_path):
            continue
        print(name)
        # reinitialize tracker, not sure if it's needed but better be safe. TODO check if it's needed
        if falsecolor:
            loader = scene.falsecolor
        else:
            loader = scene.hsi
        y_pred = [scene.init_bbox.copy()]
        tracker.initialize(loader[0], scene.init_bbox, scene.name)
        for frame_id in trange(1, len(loader)):
            img = loader[frame_id]
            bbox = tracker.pred(img)
            y_pred.append(bbox)
            
        if tracker.do_postprocessing:
            y_pred = tracker.postprocessed_pred()
        
        os.makedirs(output_dir, exist_ok=True)
        save_rects(pred_path, y_pred)
