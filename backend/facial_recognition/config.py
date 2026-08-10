"""
config.py
---------
Central configuration for the Digital Estate Assets Discovery
facial biometric verification prototype.

Keeping all tunable constants in one place makes the system easy to
extend later (e.g. swapping thresholds, model packs, or storage paths)
without touching business logic in the modules themselves.
"""

from pathlib import Path

# --------------------------------------------------------------------------
# Storage paths
# --------------------------------------------------------------------------
# Root directory where all enrolled embeddings are stored, organized by
# Beneficiary ID: data/embeddings/<beneficiary_id>/sample_XX.npy
BASE_DIR: Path = Path(__file__).resolve().parent
EMBEDDINGS_DIR: Path = BASE_DIR / "data" / "embeddings"

# --------------------------------------------------------------------------
# InsightFace model settings
# --------------------------------------------------------------------------
# "buffalo_l" bundles SCRFD (detection) + ArcFace (recognition/embedding)
# + a 106-point landmark model, all as ONNX Runtime graphs.
INSIGHTFACE_MODEL_PACK: str = "buffalo_l"

# Detection input size for SCRFD. Must be a multiple of 32.
DETECTION_SIZE: tuple[int, int] = (640, 640)

# ctx_id = -1 forces CPU execution via ONNX Runtime (safe default for a
# PoC running on any machine). Set to 0 to use the first GPU if available.
CTX_ID: int = -1

# --------------------------------------------------------------------------
# Enrollment settings
# --------------------------------------------------------------------------
# Number of face samples to capture per beneficiary during enrollment.
ENROLLMENT_SAMPLE_COUNT: int = 5

# Minimum number of frames to skip between accepted samples, so that
# consecutive samples are not near-duplicates of the same pose. This also
# doubles as the inference cadence for the live preview (see preview.py):
# SCRFD/ArcFace only run once every `ENROLLMENT_FRAME_INTERVAL` displayed
# frames, keeping the preview smooth between inference cycles.
ENROLLMENT_FRAME_INTERVAL: int = 5

# Minimum SCRFD detection confidence to accept a face during enrollment
# or verification.
MIN_DETECTION_CONFIDENCE: float = 0.5

# --------------------------------------------------------------------------
# Verification settings
# --------------------------------------------------------------------------
# Cosine similarity threshold above which a live face is considered a
# match against an enrolled embedding. ArcFace embeddings typically use
# thresholds in the 0.35-0.45 range for cosine similarity (normalized
# embeddings); tune this empirically for your camera/lighting setup.
VERIFICATION_THRESHOLD: float = 0.40

# --------------------------------------------------------------------------
# Liveness check settings
# --------------------------------------------------------------------------
# Minimum normalized horizontal displacement of the nose keypoint
# (relative to face bounding-box width) required to count as a head turn.
HEAD_TURN_DISPLACEMENT_RATIO: float = 0.08

# Eye Aspect Ratio (EAR) drop threshold used to detect a blink when
# 106-point landmarks are available.
BLINK_EAR_THRESHOLD: float = 0.21

# Max number of frames to wait for a liveness gesture before giving up.
LIVENESS_TIMEOUT_FRAMES: int = 150

# --------------------------------------------------------------------------
# Camera settings
# --------------------------------------------------------------------------
CAMERA_INDEX: int = 0
CAMERA_WIDTH: int = 1280
CAMERA_HEIGHT: int = 720

# --------------------------------------------------------------------------
# Preview / UI settings
# --------------------------------------------------------------------------
# How many displayed frames to skip between expensive SCRFD+ArcFace
# inference calls during verification's face-acquisition loop. The
# previous detection result is reused on the skipped frames so the
# bounding box/landmarks stay on screen without extra inference cost.
VERIFICATION_INFERENCE_INTERVAL: int = 3

# Max frames to search for a face during verification before giving up
# and raising NoFaceDetectedError.
VERIFICATION_SEARCH_TIMEOUT_FRAMES: int = 150

# How long (in seconds) to hold the final result screen (ENROLLMENT
# COMPLETE / VERIFIED / NOT VERIFIED) on screen before the preview closes.
RESULT_SCREEN_DURATION_SECONDS: float = 2.5

# Refresh interval for WebPreviewRenderer's background camera-stream thread
# (see modules/web_preview.py). This is what makes the verification preview
# feel like real-time video -- it's independent of how fast SCRFD/ArcFace
# inference is running. 1/30 = ~30fps.
PREVIEW_REFRESH_INTERVAL_SECONDS: float = 1 / 30
