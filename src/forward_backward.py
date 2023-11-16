import numpy as np
from tqdm import trange, tqdm
from torchdata.datapipes.iter import Prefetcher

from dataclasses import dataclass
from src.util import load_rects, draw_rect, HOTDataset, HOTDatasetMultiCam, save_rects
from src.tracker import OSTrackWrapper, clip_box
from src.HyperTools import overlap_ratio


@dataclass
class Track:
    bboxes: np.ndarray
    scores: np.ndarray 
    
    def __len__(self):
        return len(self.bboxes)

class ImageLoaderRanged:
    
    def __init__(self, image_loaders, start, end, inc):
        self.image_loaders = image_loaders
        self.start = start
        self.end = end
        self.inc = inc
        
    def __len__(self):
        return (self.end - self.start) * self.inc

    def __getitem__(self, i):
        if i >= len(self):
            raise IndexError            
        frame_id = self.start + i * self.inc
        return frame_id, self.image_loaders[0][frame_id], self.image_loaders[1][frame_id]

def get_distractions(bbox, prev_merged, score, tracks, frame_id, inc, start_frame_id, start_bbox, debug=False):
    # the track is going in the other direction
    if len(tracks) == 0:
        return []
    merged = []
    distractions = []
    for i,t in enumerate(tracks):
        tbbox = t.bboxes[frame_id]
        iou = overlap_ratio(bbox.reshape(-1,4), tbbox.reshape(-1,4))
        original_iou = overlap_ratio(
            start_bbox.reshape(-1,4),
            t.bboxes[start_frame_id].reshape(-1,4))[0]
        # calculate how well is the tbbox connected to other boxes in it's track
        prev_frame_id = frame_id - inc
        tfit = min(
            t.scores[frame_id], # current score
            t.scores[prev_frame_id]) # previous score
        if iou > 0.8:
            if score > 0.6 or i in prev_merged:
                continue
            elif tfit > 1.1 * score and original_iou < 0.2:
#                 if debug:
#                     print("iou, tfit, score, original_iou", iou, tfit, score, original_iou)
                distractions.append(i)
    return distractions

def run_ostrack(
    ostrack: OSTrackWrapper,
    dim_reduction, camera_type, # new addition
    frames, init_bbox, start, end,
    known_tracks, debug=False
):
    if end > start:
        desc = "forward"
        known_tracks = known_tracks["backward"]
        inc = 1
    else:
        desc = "backward"
        inc = -1
        known_tracks = known_tracks["forward"]
        
    frames_range = ImageLoaderRanged(frames, start, end, inc)
    
    for i, (frame_id, hsi, falsecolor) in enumerate(tqdm(Prefetcher(frames_range), desc=desc)):
        if i == 0:
            h,w = falsecolor.shape[:2]
            init_bbox = np.array(clip_box(init_bbox, h, w, margin=5))
            bbox = init_bbox
            dim_reduction.initialize(hsi, bbox, camera_type) # mew addition
            init_hsi_reduced = dim_reduction(hsi, bbox)
            init_falsecolor = falsecolor.copy()
            track = [bbox]
            track_scores = [1.0]
            if debug:
                init_frame = falsecolor.copy()
            merged = []
        else:
            bbox = np.array(clip_box(bbox, h, w, margin=5))
            hsi_reduced = dim_reduction(hsi, bbox)  # new addition
            ostrack.initialize(init_hsi_reduced, init_bbox)
            candidates, scores, _ = ostrack.track(hsi_reduced, bbox, 0)
            candidates = candidates.numpy(force=True)
            scores = scores.numpy(force=True).flatten()
            ostrack.initialize(init_falsecolor, init_bbox)
            candidates2, scores2, _ = ostrack.track(falsecolor, bbox, 0)
            candidates2 = candidates2.numpy(force=True)
            scores2 = scores2.numpy(force=True).flatten()
            # combine predictions pick the tracker with higher score
            candidates = np.concatenate((candidates, candidates2))
            scores = np.concatenate((0.9 * scores, scores2))
            
            n_pred = len(candidates)
            if len(known_tracks) > 0:
                candidates = np.concatenate((candidates, [t.bboxes[frame_id] for t in known_tracks]))
                scores = np.concatenate((scores, np.full(len(known_tracks), 0.2)))
            for j, (c, s) in enumerate(zip(candidates, scores)):
                distr = get_distractions(c, merged, s, known_tracks, frame_id, inc, start, init_bbox, debug)
                if len(distr) > 0:
                    scores[j] = 0
                    distr_id = distr[0]
#                     if debug:
#                         print("frame id", frame_id)
#                         print(f"distractor from track {j}")
#                         fig, axs = plt.subplots(1,2)
#                         init_frame_copy = draw_rect(init_frame.copy(), init_bbox, color=(0,255,0))
#                         init_frame_copy = draw_rect(init_frame_copy, known_tracks[distr_id].bboxes[start], color=(255,0,0))
#                         axs[0].imshow(init_frame_copy)
#                         frame_copy = draw_rect(frame.copy(), c)
#                         axs[1].imshow(frame_copy)
#                         plt.show()
                        
            j = np.argmax(scores)
            if scores[j] > 0:
                bbox, score = candidates[j], scores[j]
                merged = []
                for k,t in enumerate(known_tracks):
                    if overlap_ratio(t.bboxes[frame_id].reshape(-1,4), bbox.reshape(-1,4)) > 0.8:
                        merged.append(k)
            else:
                bbox, score = bbox, 0
                merged = []
            track.append(bbox)
            track_scores.append(score)
    return Track(np.array(track, dtype=int), np.array(track_scores))


def backward_forward(
    ostrack: OSTrackWrapper, dim_reduction,
    frames, bbox_init, camera_type, scene_name, debug=False, iou_thrsh=0.2, maxiter=3
):
    if debug:
        out_dir = f"../outputs/forward_backward-neighbour/{camera_type}/{scene_name}"
        os.makedirs(out_dir, exist_ok=True)
    niter = 0
    start = 0
    end = len(frames[0])
    tracks = {"forward": [], "backward": []}
    dbg_backward = None
    print("niter: [start, end]")
    while start + 1 < end and niter < maxiter:
        print(f"{niter}: [{start},{end}]")
        niter += 1
        # run forward
        forward = run_ostrack(
            ostrack, dim_reduction, camera_type,
            frames, bbox_init, start, end, tracks, debug
        )
        if len(tracks["forward"]) > 0:
            # pad forward pass with the previous one
            forward.bboxes = np.concatenate((tracks["forward"][-1].bboxes[:start], forward.bboxes))
            forward.scores = np.concatenate((np.ones(start), forward.scores))
        tracks["forward"].append(forward)
        if debug:
            save_rects(os.path.join(out_dir, f"forward_{niter}.txt"), forward.bboxes)
            
        if len(forward) != len(frames[0]):
            print(len(forward), len(frames[0]))
            assert False
        
        # bisect until backtracking is consistent
        consistent = False
        _start = start
        _end = end
        _backtracking_runs = 0
        while not consistent and niter < maxiter:
            _backtracking_runs += 1
            if debug:
                print("consistency check on:", _start, _end)
            backward = run_ostrack(
                ostrack, dim_reduction, camera_type,
                frames, forward.bboxes[_end-1], _end - 1, _start - 1, tracks, debug)
            backward.bboxes = backward.bboxes[::-1] # reverse
            backward.scores = backward.scores[::-1] # reverse
            # pad backward
            backward.bboxes = np.concatenate((
                np.zeros_like(forward.bboxes[:_start]),
                backward.bboxes,
                forward.bboxes[_end:]
            ))
            backward.scores = np.concatenate((
                np.zeros_like(forward.scores[:_start]),
                backward.scores,
                forward.scores[_end:]
            ))
            if len(backward) != len(frames[0]) or len(backward.bboxes) != len(backward.scores):
                print(len(forward), len(frames[0]), len(backward), len(backward.bboxes), len(backward.scores))
                assert False
            tracks["backward"].append(backward)
            if dbg_backward is None:
                dbg_backward = backward.bboxes.copy()
                if debug:
                    save_rects(os.path.join(out_dir, f"backward_0.txt"), dbg_backward)
            
            iou = overlap_ratio(forward.bboxes[_start:_end], backward.bboxes[_start:_end])
            if debug:
                plt.plot(np.arange(_start, _end), iou, label=str(_backtracking_runs))
            idx = np.argwhere(iou < iou_thrsh) # inconsistency check
            idx += _start
            if len(idx) == 0:
                consistent = True
                continue
            last_consistent = np.max(idx)
            _end = (last_consistent + _start) // 2 # bug fix
#             _end = last_consistent - 1
            if _start + 1 >= _end:
                break

        if not consistent:
            if debug:
                print("failure to find consistent backtracking")
            break
        else:
            if debug:
                print("success, consistent range: ", _start, _end)
            dbg_backward[_start:_end] = backward.bboxes[_start:_end]
            if debug:
                save_rects(os.path.join(out_dir, f"backward_{niter}.txt"), dbg_backward)
            start = _end - 1
            bbox_init = forward.bboxes[start]
    
        if debug:
            plt.ylim(0,1)
            plt.legend()
            plt.show()
    return tracks["forward"][-1].bboxes
