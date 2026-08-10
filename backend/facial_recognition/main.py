"""
main.py
-------
CLI entry point for the Digital Estate Assets Discovery facial biometric
verification prototype.

This ties the modules together but contains no business logic itself —
all real work happens inside modules/. This keeps main.py trivial to
replace later with a REST API, CLI framework (e.g. Typer/Click), or
frontend integration.

Usage:
    python main.py enroll <beneficiary_id>
    python main.py verify <beneficiary_id>
"""

from __future__ import annotations

import sys

from backend.facial_recognition.modules.enrollment import FaceEnrollment
from backend.facial_recognition.modules.exceptions import (
    BeneficiaryNotEnrolledError,
    CameraUnavailableError,
    LivenessCheckFailedError,
    MultipleFacesDetectedError,
    NoFaceDetectedError,
)
from backend.facial_recognition.modules.face_detector import FaceDetector
from backend.facial_recognition.modules.liveness import LivenessCheck
from backend.facial_recognition.modules.verification import FaceVerification


def print_usage() -> None:
    print("Usage:")
    print("  python main.py enroll <beneficiary_id>")
    print("  python main.py verify <beneficiary_id>")


def run_enrollment(beneficiary_id: str) -> None:
    """Build dependencies and run the enrollment flow for a beneficiary."""
    detector = FaceDetector()
    liveness_check = LivenessCheck()
    enrollment = FaceEnrollment(detector=detector, liveness_check=liveness_check)
    enrollment.enroll(beneficiary_id)


def run_verification(beneficiary_id: str) -> None:
    """Build dependencies and run the verification flow for a beneficiary."""
    detector = FaceDetector()
    liveness_check = LivenessCheck()
    verification = FaceVerification(detector=detector, liveness_check=liveness_check)
    result = verification.verify(beneficiary_id)

    if result.verified:
        print(f"RESULT: VERIFIED (score={result.similarity_score:.4f})")
    else:
        print(f"RESULT: NOT VERIFIED (score={result.similarity_score:.4f})")


def main() -> None:
    if len(sys.argv) != 3:
        print_usage()
        sys.exit(1)

    command, beneficiary_id = sys.argv[1].lower(), sys.argv[2]

    try:
        if command == "enroll":
            run_enrollment(beneficiary_id)
        elif command == "verify":
            run_verification(beneficiary_id)
        else:
            print_usage()
            sys.exit(1)

    # --- Common, expected error conditions get clean, informative output ---
    except CameraUnavailableError as e:
        print(f"[ERROR] Camera problem: {e}")
        sys.exit(2)
    except NoFaceDetectedError as e:
        print(f"[ERROR] No face detected: {e}")
        sys.exit(3)
    except MultipleFacesDetectedError as e:
        print(f"[ERROR] Multiple faces detected: {e}")
        sys.exit(4)
    except LivenessCheckFailedError as e:
        print(f"[ERROR] Liveness check failed: {e}")
        sys.exit(5)
    except BeneficiaryNotEnrolledError as e:
        print(f"[ERROR] Beneficiary not enrolled: {e}")
        sys.exit(6)


if __name__ == "__main__":
    main()
