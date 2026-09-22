"""
test_dog_detector.py

Usage:
    python test_dog_detector.py path/to/image.jpg

This tests the dog-vs-non-dog gate WITHOUT loading TensorFlow/emotion models.
It is the fastest way to verify that humans are rejected before the emotion model.
"""
import sys
from pathlib import Path
import cv2

from dog_detector import detect_dog


def main():
    if len(sys.argv) != 2:
        print("Usage: python test_dog_detector.py <image.jpg>")
        raise SystemExit(2)

    path = Path(sys.argv[1])
    frame = cv2.imread(str(path))
    if frame is None:
        print(f"Could not read image: {path}")
        raise SystemExit(1)

    result = detect_dog(frame)
    print("\nDOG GATE RESULT")
    print(f"available  : {result.get('available')}")
    print(f"present    : {result.get('present')}")
    print(f"confidence : {result.get('confidence'):.4f}")
    print(f"bbox       : {result.get('bbox')}")
    if result.get('error'):
        print(f"error      : {result['error']}")

    if result.get('present'):
        print("\nRESULT: DOG DETECTED -> emotion model may run.")
    else:
        print("\nRESULT: NO DOG -> emotion model must NOT run.")


if __name__ == '__main__':
    main()
