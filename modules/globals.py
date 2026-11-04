# --- START OF FILE globals.py ---

import os
from typing import List, Dict, Any

ROOT_DIR = os.path.dirname(os.path.abspath(__file__))
WORKFLOW_DIR = os.path.join(ROOT_DIR, "workflow")

# Canonical media extensions, defined once so the file dialogs and
# has_image_extension never drift. GIF is intentionally excluded: OpenCV's
# cv2.imread/imwrite (the only image I/O this app uses) cannot decode or
# encode GIF on 4.10 or 4.11, so offering it would silently fail. WEBP works
# via the libwebp bundled with opencv-python.
IMAGE_EXTENSIONS = (".png", ".jpg", ".jpeg", ".bmp", ".webp")
VIDEO_EXTENSIONS = (".mp4", ".mkv")

# Face Mapping Data
source_target_map: List[Dict[str, Any]] = [] # Stores detailed map for image/video processing
simple_map: Dict[str, Any] = {}             # Stores simplified map (embeddings/faces) for live/simple mode

# Paths
source_path: str | None = None
target_path: str | None = None
output_path: str | None = None

# Processing Options
frame_processors: List[str] = []
keep_fps: bool = True
keep_audio: bool = True
keep_frames: bool = False
many_faces: bool = False         # Process all detected faces with default source
map_faces: bool = False          # Use source_target_map or simple_map for specific swaps
poisson_blend: bool = False      # Enable Poisson Blending for smoother face swaps
color_correction: bool = False   # Enable color correction (implementation specific)
nsfw_filter: bool = False

# Video Output Options
video_encoder: str | None = None
video_quality: int | None = None # Typically a CRF value or bitrate

# Live Mode Options
live_mirror: bool = False
live_resizable: bool = True
camera_input_combobox: Any | None = None # Placeholder for UI element if needed
webcam_preview_running: bool = False
show_fps: bool = False

# System Configuration
max_memory: int | None = None        # Memory limit in GB? (Needs clarification)
execution_providers: List[str] = []  # e.g., ['CUDAExecutionProvider', 'CPUExecutionProvider']
execution_threads: int | None = None # Number of threads for CPU execution
headless: bool | None = None         # Run without UI?
log_level: str = "error"             # Logging level (e.g., 'debug', 'info', 'warning', 'error')

# Face Processor UI Toggles (Example)
fp_ui: Dict[str, bool] = {"face_enhancer": False, "face_enhancer_gpen256": False, "face_enhancer_gpen512": False}

# Face Swapper Specific Options
face_swapper_enabled: bool = True # General toggle for the swapper processor
opacity: float = 1.0              # Blend factor for the swapped face (0.0-1.0)
sharpness: float = 0.0            # Sharpness enhancement for swapped face (0.0-1.0+)

# Mouth Mask Options
mouth_mask: bool = False           # Enable mouth area masking/pasting
show_mouth_mask_box: bool = False  # Visualize the mouth mask area (for debugging)
mask_feather_ratio: int = 12       # Denominator for feathering calculation (higher = smaller feather)
mask_down_size: float = 0.1        # Expansion factor for lower lip mask (relative)
mask_size: float = 1.0             # Expansion factor for upper lip mask (relative)
mouth_mask_size: float = 0.0       # Mouth mask size (0-100; 0=off, 100=mouth to chin)

# --- START: Added for Frame Interpolation ---
enable_interpolation: bool = True # Toggle temporal smoothing
interpolation_weight: float = 0  # Blend weight for current frame (0.0-1.0). Lower=smoother.
# --- END: Added for Frame Interpolation ---

# Face detection resolution (160, 320, or 640).
# Lower = faster detection, fewer FLOPs, less accurate at distance.
# Changes require face analyser re-init (handled in UI by clearing FACE_ANALYSER).
# DEFAULT_DET_SIZE is the single source of truth for the default — UI,
# face_analyser, and CLI all reference it so they can't drift apart.
DEFAULT_DET_SIZE: int = 640
det_size: int = DEFAULT_DET_SIZE

# Webcam capture resolution requested via cv2.CAP_PROP_FRAME_WIDTH/HEIGHT.
# Camera may negotiate to its nearest supported size — actual size is
# printed in console as "[VideoCapturer] WxH @ FPS".
# Default 640x480: native on virtually every webcam, fastest reliable mode
# on USB 2.0. Higher 16:9 tiers (540/720/1080) selectable in UI.
capture_resolution: tuple = (640, 480)

# --- END OF FILE globals.py ---

import threading
dml_lock = threading.Lock()

# ─── Extended settings (Settings dialog). Every new feature defaults OFF. ───
PROJECT_DIR = os.path.dirname(ROOT_DIR)

# Live face detection frequency: run the detector every N frames.
# 0 = auto (legacy formula: max(1, round(camera_fps * 0.04))).
det_interval: int = 0

# Face Tracking 2.0 — master switch. Off = legacy "hold last detection".
tracking_v2: bool = False
tracking_optical_flow: bool = True     # LK optical flow between detections
tracking_smoothing: bool = True        # One Euro filter on keypoints
tracking_smoothing_strength: float = 0.5  # 0 = raw, 1 = very smooth
tracking_grace_frames: int = 6         # hold a lost face this many frames
tracking_max_motion: float = 0.25      # max per-frame shift, fraction of face size
tracking_max_rotation: float = 20.0    # max per-frame roll change, degrees
tracking_min_confidence: float = 0.5   # below this a detection counts as lost
tracking_auto_redetect: bool = True    # force detection next frame when tracking fails

# Freeze the output on the last swapped frame when the face is lost,
# so the real face is never shown.
freeze_on_face_lost: bool = False

# Realism (all off by default).
adaptive_face_mask: bool = False       # landmark-shaped blend mask
adaptive_mask_feather: float = 0.5     # 0..1 feather amount
color_matching: bool = False           # LAB mean/std transfer to target skin
color_matching_strength: float = 0.7
temporal_stabilization: bool = False   # motion-aware blend of swap output
temporal_strength: float = 0.5
eye_preservation: bool = False         # blend original eyes (natural blinking)
eye_preservation_strength: float = 0.4
mouth_preservation: bool = False       # blend original lips (natural speech)
mouth_preservation_strength: float = 0.5
dynamic_enhancement: bool = False      # run face enhancer only when useful
dynamic_enhancement_min_face: int = 110   # px; smaller faces skip enhancer
dynamic_enhancement_max_motion: float = 0.08  # fraction of face size per frame

# TensorRT (experimental). Applied on restart: sessions are built at load time.
use_tensorrt: bool = False
trt_fp16: bool = True
trt_builder_optimization_level: int = 3   # 0..5, higher = slower build, faster engine
trt_max_workspace_gb: int = 4
trt_timing_cache: bool = True
trt_cache_dir: str = os.path.join(PROJECT_DIR, "trt_cache")

# ─── Swap model, face parts, occlusion, output (Settings dialog) ───
# Face swap model key (modules/processors/frame/swapper_models.py).
swapper_model: str = "inswapper_128"
# Face-part masks (BiSeNet). Off = whole face swapped as before.
face_parts_mask: bool = False
face_parts: Dict[str, bool] = {
    "forehead": True, "brows": True, "eyes": True, "nose": True,
    "cheeks": True, "mouth": True, "chin": True,
}
face_parts_feather: float = 0.5
# Occlusion mask (XSeg): hands / mic / objects in front of the face stay real.
occlusion_mask: bool = False
# Use the swap model's own face mask (HyperSwap outputs one).
use_model_mask: bool = False
# Recompute face-part / occlusion masks every N frames (1 = every frame).
mask_interval: int = 1
# Camera capture frame rate requested from the device.
capture_fps: int = 60
# Virtual camera output (pyvirtualcam -> OBS Virtual Camera).
virtual_camera: bool = False
virtual_camera_resolution: tuple = (0, 0)   # (0, 0) = same as the processed frame

# ─── Multi-photo source, pipeline, reprojection (Settings dialog) ───
# Several photos of the same person, blended by the target's head pose.
multi_source: bool = False
source_paths_extra: List[str] = []
# Live: run face detection in its own thread, parallel to the swap.
async_detection: bool = False
# Live: move the last finished swap onto the newest camera frame (low latency).
reprojection: bool = False

# Grain & sharpness matching to the camera image (Settings -> Realism).
grain_matching: bool = False
grain_amount: float = 1.0            # share of the missing noise added (0..1.5)
grain_sharpness_amount: float = 0.7  # how far detail is pulled to the camera's (0..1)

# ─── Faces: fader, profile fade, library; upscaler; frame interpolation ───
# Swap on/off switch (main window button). Always on unless the user turns it off.
swap_enabled: bool = True
# Face Fader: ramp swap on/off and morph between identities over N seconds.
face_fader: bool = False
fader_seconds: float = 1.5
# Profile fade: lower the swap past these head angles (degrees, from 1k3d68).
profile_fade: bool = False
profile_yaw_start: float = 35.0
profile_yaw_end: float = 60.0
profile_pitch_start: float = 25.0
profile_pitch_end: float = 45.0
# Super-resolution of the output (x2). mode: "frame" = whole frame, "face" = face only.
upscaler: bool = False
upscaler_mode: str = "face"
upscaler_model: str = "span"
# Live frame interpolation (RIFE): doubles the output frame rate, +half a frame of latency.
frame_interpolation: bool = False

# Ask GitHub for a newer Deep-Boost-Cam release at start (on by default).
check_updates: bool = True
