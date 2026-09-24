"""
test_pose_image.py

Diagnose pose-keypoint quality on a SINGLE image -- no live dog, webcam,
or video needed. Draws every detected keypoint with its index number so
you can see whether withers (22) and the paw points (0, 3, 6, 9) are
actually being picked up.

Usage:
    python test_pose_image.py path/to/dog_photo.jpg
"""

import sys
import cv2
import numpy as np

from pose_motion import load_pose_model, extract_keypoints, classify_posture, KP


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_pose_image.py <image.jpg>")
        raise SystemExit(2)

    path = sys.argv[1]
    frame = cv2.imread(path)
    if frame is None:
        print(f"Could not read image: {path}")
        raise SystemExit(1)

    print("Loading pose model...")
    pose_model = load_pose_model()
    if pose_model is None:
        print("ERROR: no pose model found at models/dog_pose_best.pt")
        raise SystemExit(1)

    kpts, bbox_diagonal, pose_confidence = extract_keypoints(pose_model, frame)

    print("\n" + "=" * 60)
    print(f"YOLO pose-box confidence : {pose_confidence:.3f}")

    if kpts is None:
        print("Keypoints                : NONE")
        print("\nThe pose model did not return usable keypoints for this")
        print("image at all (failed the confidence/valid-keypoint gate in")
        print("extract_keypoints()). Try a clearer, closer, full-body shot,")
        print("or lower MIN_POSE_CONFIDENCE / MIN_VALID_KEYPOINTS in")
        print("pose_motion.py to see raw output.")
        print("=" * 60)
        return

    names = list(KP.keys())
    print(f"Bounding-box diagonal    : {bbox_diagonal:.1f}px")
    print("\nPer-keypoint status:")
    for name in names:
        x, y = kpts[KP[name]]
        valid = not (abs(x) < 1e-6 and abs(y) < 1e-6)
        flag = "OK " if valid else "MISSING"
        marker = "  <-- withers" if name == "withers" else (
            "  <-- paw" if "paw" in name else "")
        print(f"  [{KP[name]:2d}] {name:<18s} {flag}  ({x:.0f}, {y:.0f}){marker}")

    posture = classify_posture(kpts, bbox_diagonal)
    print(f"\nPosture result           : {posture['label']} (confidence {posture['confidence']:.2f})")
    print("=" * 60)

    # Draw an annotated copy so you can see it visually too.
    annotated = frame.copy()
    for name in names:
        x, y = int(kpts[KP[name]][0]), int(kpts[KP[name]][1])
        if x <= 0 or y <= 0:
            continue
        color = (0, 0, 255) if name == "withers" else (
            (255, 0, 0) if "paw" in name else (0, 255, 0))
        cv2.circle(annotated, (x, y), 5, color, -1)
        cv2.putText(annotated, str(KP[name]), (x + 6, y - 6),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.45, (255, 255, 255), 1)

    out_path = "pose_check_output.jpg"
    cv2.imwrite(out_path, annotated)
    print(f"\nAnnotated image saved to: {out_path}")
    print("Red = withers, Blue = paws, Green = everything else.")


if __name__ == "__main__":
    main()
