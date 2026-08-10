"""
exceptions.py
-------------
Custom exception types for the facial biometric verification system.

Using typed exceptions (instead of generic RuntimeError/ValueError)
lets calling code — and later, an API layer — distinguish between
different failure modes and react appropriately.
"""


class CameraUnavailableError(Exception):
    """Raised when the webcam cannot be opened or read from."""


class NoFaceDetectedError(Exception):
    """Raised when zero faces are found in a frame that requires one."""


class MultipleFacesDetectedError(Exception):
    """Raised when more than one face is found in a frame that requires exactly one."""


class LivenessCheckFailedError(Exception):
    """Raised when the liveness gesture (blink / head turn) is not detected in time."""


class BeneficiaryNotEnrolledError(Exception):
    """Raised when verification is attempted for a Beneficiary ID with no stored embeddings."""
