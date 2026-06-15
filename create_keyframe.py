import cv2
import numpy as np
import os
import concurrent.futures
from tqdm import tqdm

# --- Configuration parameters ---
TARGET_FRAMES = 3          # target number of frames
MIN_TIME_DISTANCE = 2.0    # minimum time gap (seconds); prevents filler frames from being too close to already-selected ones.
SAMPLE_STEP = 5            # sampling stride (compute variance every N frames; smaller is slower but more accurate)
SCENE_THRESHOLD = 30.0     # shot-detection threshold (inter-frame grayscale mean difference; larger is less sensitive)
ENTROPY_WEIGHT = 0.3       # weight of color entropy in the joint score
ENTROPY_BINS = 32          # number of bins in the color histogram


def calculate_quality(image, gray=None):
    """
    Joint quality score = sharpness x (1 + w x color_entropy)

    - sharpness: variance of the Laplacian, measuring texture richness.
    - color entropy: information entropy of the HSV H-channel histogram, measuring color diversity.
      Pure-text frames / black frames / monochrome frames have very low entropy and are filtered out effectively.

    Args:
        image: numpy array in BGR format.
        gray:  if the grayscale image was already computed externally, pass it in to avoid recomputing.
    """
    if image is None:
        return 0.0
    if gray is None:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # sharpness (variance of the Laplacian)
    sharpness = cv2.Laplacian(gray, cv2.CV_64F).var()

    # color entropy (HSV H channel)
    hsv = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    hist = cv2.calcHist([hsv], [0], None, [ENTROPY_BINS], [0, 180])
    hist = hist.flatten() / (hist.sum() + 1e-8)  # normalize
    nonzero = hist[hist > 0]
    entropy = -np.sum(nonzero * np.log2(nonzero)) if len(nonzero) > 0 else 0.0

    # joint score
    return sharpness * (1.0 + ENTROPY_WEIGHT * entropy)


def is_time_conflict(new_timestamp, selected_frames, min_dist):
    """Check whether the new frame's timestamp is too close to any already-selected frame."""
    for frame_data in selected_frames:
        if abs(new_timestamp - frame_data['timestamp']) < min_dist:
            return True
    return False


def extract_hybrid_keyframes(video_path):
    """
    Smart hybrid strategy (v3 - single-decode version):

    In a single decode pass over the video, simultaneously perform:
      a) shot-cut detection via inter-frame difference (replaces PySceneDetect, eliminating a second decode pass)
      b) joint quality scoring (sharpness + color entropy)
      c) candidate-frame pool construction

    Then run a two-stage frame selection:
      1. Prioritize shot coverage (Diversity)
      2. Allocate remaining slots to the highest-quality frames (Quality)
    """
    candidate_pool = []  # stores info for every sampled frame

    try:
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            print(f"[Error] cannot open video: {video_path}")
            return []

        fps = cap.get(cv2.CAP_PROP_FPS)
        if fps <= 0:
            fps = 25.0
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        if total_frames <= 0:
            cap.release()
            return []

        # --- Single pass: shot detection + quality scoring + candidate pool construction ---
        prev_gray = None       # grayscale of the previous sampled frame, used for inter-frame difference
        scene_cuts = [0]       # list of shot-cut points (frame indices); the first frame is always the start of the first shot
        current_frame_idx = 0

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            # only compute at sampling points (non-sampled frames just advance the decode pointer)
            if current_frame_idx % SAMPLE_STEP == 0:
                gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)

                # (a) shot-cut detection: inter-frame grayscale mean difference
                if prev_gray is not None:
                    diff = cv2.absdiff(gray, prev_gray).mean()
                    if diff > SCENE_THRESHOLD:
                        scene_cuts.append(current_frame_idx)

                # (b) joint quality scoring (reuse the already-computed gray to avoid a second BGR->Gray conversion)
                score = calculate_quality(frame, gray=gray)
                timestamp = current_frame_idx / fps

                candidate_pool.append({
                    'image': frame,
                    'score': score,
                    'timestamp': timestamp,
                    'frame_idx': current_frame_idx,
                    'scene_idx': -1,  # assigned later
                })

                prev_gray = gray

            current_frame_idx += 1

        cap.release()

        if not candidate_pool:
            return []

        # --- Build shot intervals [(start, end), ...] ---
        scene_cuts.append(total_frames)  # sentinel: end point of the last shot
        scene_list = [
            (scene_cuts[i], scene_cuts[i + 1])
            for i in range(len(scene_cuts) - 1)
        ]

        # --- Annotate each candidate frame with the shot index it belongs to ---
        # use binary search for speed (scene_cuts is already sorted)
        for cand in candidate_pool:
            fidx = cand['frame_idx']
            lo, hi = 0, len(scene_list) - 1
            while lo < hi:
                mid = (lo + hi + 1) // 2
                if scene_list[mid][0] <= fidx:
                    lo = mid
                else:
                    hi = mid - 1
            cand['scene_idx'] = lo

        # --- C. Frame-selection logic ---
        selected_candidates = []

        # Step 1: pick the single best frame within each shot (best of each shot)
        # group by shot
        scenes_candidates = {}
        for cand in candidate_pool:
            s_idx = cand['scene_idx']
            if s_idx not in scenes_candidates:
                scenes_candidates[s_idx] = []
            scenes_candidates[s_idx].append(cand)

        # iterate over each shot and take the highest-scoring frame
        sorted_scene_indices = sorted(scenes_candidates.keys())

        # while slots remain and there are still unvisited shots, prefer picking from new shots
        for s_idx in sorted_scene_indices:
            if len(selected_candidates) >= TARGET_FRAMES:
                break

            # find the highest-scoring frame in this shot
            best_in_scene = max(scenes_candidates[s_idx], key=lambda x: x['score'])
            selected_candidates.append(best_in_scene)

        # Step 2: if slots are not yet full (e.g. only 1 shot, 1 frame picked, 2 frames short)
        # pick from all remaining candidates the highest-scoring ones that do not conflict in time
        while len(selected_candidates) < TARGET_FRAMES:
            # exclude already-selected frames
            selected_indices = {x['frame_idx'] for x in selected_candidates}
            remaining_pool = [c for c in candidate_pool if c['frame_idx'] not in selected_indices]

            if not remaining_pool:
                break  # no frames left to pick

            # sort by score from high to low
            remaining_pool.sort(key=lambda x: x['score'], reverse=True)

            found_valid = False
            for cand in remaining_pool:
                # check time conflict
                if not is_time_conflict(cand['timestamp'], selected_candidates, MIN_TIME_DISTANCE):
                    selected_candidates.append(cand)
                    found_valid = True
                    break  # found the best frame for this round, break and move on

            if not found_valid:
                # if all high-scoring frames are too close to selected ones, relax the distance and force-fill the top frame
                print(f"Warning: cannot satisfy minimum time gap {MIN_TIME_DISTANCE}s; force-filling the top-scoring frame.")
                selected_candidates.append(remaining_pool[0])

        # --- D. Final ordering ---
        # must output in temporal order; this is crucial for multimodal models to understand causal relations
        selected_candidates.sort(key=lambda x: x['frame_idx'])

        return [x['image'] for x in selected_candidates]

    except Exception as e:
        print(f"Error: {e}")
        return []


def batch_process(video_dir):
    """Batch-processing entry point."""
    output_dir = os.path.join(video_dir, "keyframe_hybridv2")
    if not os.path.exists(output_dir):
        os.makedirs(output_dir)

    video_files = [f for f in os.listdir(video_dir) if f.lower().endswith('.mp4')]

    # resume support: skip videos that already have output
    pending_files = []
    for file_name in video_files:
        base_name = os.path.splitext(file_name)[0]
        if os.path.exists(os.path.join(output_dir, f"{base_name}_0.jpg")):
            continue
        pending_files.append(file_name)

    if len(pending_files) < len(video_files):
        print(f"Skipping {len(video_files) - len(pending_files)} already-processed videos, {len(pending_files)} remaining")

    print(f"Start processing {len(pending_files)} videos (v3 single-decode + joint quality scoring)...")

    for i, file_name in enumerate(pending_files):
        print(f"[{i+1}/{len(pending_files)}] {file_name} ...")
        frames = extract_hybrid_keyframes(os.path.join(video_dir, file_name))

        base_name = os.path.splitext(file_name)[0]
        for idx, frame in enumerate(frames):
            cv2.imwrite(os.path.join(output_dir, f"{base_name}_{idx}.jpg"), frame)


def process_single_directory(args):
    """Process a single subdirectory; returns (dir_name, success, error_message).
    Note: defined at module top level so it can be pickled for multiprocessing.
    """
    root_dir, dir_name = args
    full_dir_path = os.path.join(root_dir, dir_name)
    try:
        batch_process(full_dir_path)
        return dir_name, True, None
    except Exception as e:
        return dir_name, False, str(e)


if __name__ == "__main__":
    # 1. set the root directory of the clips
    ROOT_DIR = r"/path/to/data/M3-Bench/videos/web/clips"

    if not os.path.exists(ROOT_DIR):
        print(f"[Error] root directory not found: {ROOT_DIR}")
        exit()

    # 2. collect all subdirectories under this path (i.e. the concrete video folders, e.g. _eJsOYC8SVU)
    all_items = os.listdir(ROOT_DIR)
    sub_directories = [d for d in all_items if os.path.isdir(os.path.join(ROOT_DIR, d))]

    total_dirs = len(sub_directories)
    print(f"Start full processing (v3: single-decode + joint quality scoring)")
    print(f"Root directory: {ROOT_DIR}")
    print(f"Found {total_dirs} subdirectories to process")
    print("=" * 60)

    # 3. use multiprocessing to process multiple subdirectories in parallel, with a progress bar
    max_workers = 8  # min(total_dirs, os.cpu_count() or 4)
    print(f"Using {max_workers} parallel worker processes")
    print("=" * 60)

    success_count = 0
    failed_dirs = []

    # build the argument list; each element is a (ROOT_DIR, dir_name) tuple
    task_args = [(ROOT_DIR, d) for d in sub_directories]

    with concurrent.futures.ProcessPoolExecutor(max_workers=max_workers) as executor:
        # submit all tasks
        future_to_dir = {
            executor.submit(process_single_directory, args): args[1]
            for args in task_args
        }

        # show progress with tqdm
        with tqdm(total=total_dirs, desc="Directory progress", unit="dir") as pbar:
            for future in concurrent.futures.as_completed(future_to_dir):
                dir_name, success, error = future.result()
                if success:
                    success_count += 1
                    pbar.set_postfix_str(f"OK {dir_name}")
                else:
                    failed_dirs.append((dir_name, error))
                    pbar.set_postfix_str(f"FAIL {dir_name}")
                pbar.update(1)

    # 5. print final statistics
    print("\n" + "=" * 60)
    print(f"All tasks finished.")
    print(f"Success: {success_count}/{total_dirs} directories")
    if failed_dirs:
        print(f"Failed: {len(failed_dirs)} directories")
        for dir_name, error in failed_dirs:
            print(f"   - {dir_name}: {error}")
    else:
        print(f"All succeeded.")
