import cv2
import os
import time

from pose_motion import (
    load_pose_model,
    extract_keypoints,
    PoseStateTracker,
)


# ============================================================
# CONFIG
# ============================================================

VIDEO_PATH = "test.mp4"

# For webcam:
# VIDEO_PATH = 0

WINDOW_NAME = "Dog Pose + Motion Test"


# ============================================================
# DRAW KEYPOINTS
# ============================================================

def draw_keypoints(frame, kpts):

    if kpts is None:
        return frame

    for i, (x, y) in enumerate(kpts):

        x = int(x)
        y = int(y)

        # Ignore invalid points
        if x <= 0 or y <= 0:
            continue

        cv2.circle(
            frame,
            (x, y),
            4,
            (0, 255, 0),
            -1
        )

        cv2.putText(
            frame,
            str(i),
            (x + 5, y - 5),
            cv2.FONT_HERSHEY_SIMPLEX,
            0.4,
            (255, 255, 255),
            1
        )

    return frame


# ============================================================
# DRAW INFORMATION PANEL
# ============================================================

def draw_info(
    frame,
    state,
    pose_confidence
):

    motion = state["motion"]
    posture = state["posture"]
    tail_low = state["tail"]["low"]

    # --------------------------------------------------------
    # Panel
    # --------------------------------------------------------

    cv2.rectangle(
        frame,
        (10, 10),
        (390, 205),
        (0, 0, 0),
        -1
    )

    # --------------------------------------------------------
    # Motion
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Motion: {motion['label']}",
        (20, 42),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 255),
        2
    )

    # --------------------------------------------------------
    # Motion score
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Motion Score: {motion['score']:.4f}",
        (20, 70),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.55,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Posture
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Posture: {posture['label']}",
        (20, 105),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.75,
        (0, 255, 0),
        2
    )

    # --------------------------------------------------------
    # Posture confidence
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Posture Conf: {posture['confidence']:.2f}",
        (20, 133),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    # --------------------------------------------------------
    # Tail
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"Tail Low: {'YES' if tail_low else 'NO'}",
        (20, 165),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.7,
        (255, 200, 0),
        2
    )

    # --------------------------------------------------------
    # Pose confidence
    # --------------------------------------------------------

    cv2.putText(
        frame,
        f"YOLO Conf: {pose_confidence:.2f}",
        (20, 193),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (255, 255, 255),
        1
    )

    return frame


# ============================================================
# MAIN
# ============================================================

def main():

    print("=" * 65)
    print("DOG POSE + MOTION TEST")
    print("=" * 65)

    # --------------------------------------------------------
    # Load pose model
    # --------------------------------------------------------

    print("\nLoading pose model...")

    pose_model = load_pose_model()

    if pose_model is None:

        print("\nERROR: Pose model was not found.")

        print(
            "Expected model:"
        )

        print(
            os.path.abspath(
                "models/dog_pose_best.pt"
            )
        )

        return

    print(
        "\nPose model loaded successfully."
    )

    # --------------------------------------------------------
    # Open video
    # --------------------------------------------------------

    print(
        f"\nOpening video: {VIDEO_PATH}"
    )

    cap = cv2.VideoCapture(
        VIDEO_PATH
    )

    if not cap.isOpened():

        print(
            f"\nERROR: Could not open video:"
            f" {VIDEO_PATH}"
        )

        return

    # --------------------------------------------------------
    # Create temporal tracker
    # --------------------------------------------------------

    tracker = PoseStateTracker()

    # --------------------------------------------------------
    # Video information
    # --------------------------------------------------------

    fps = cap.get(
        cv2.CAP_PROP_FPS
    )

    width = int(
        cap.get(
            cv2.CAP_PROP_FRAME_WIDTH
        )
    )

    height = int(
        cap.get(
            cv2.CAP_PROP_FRAME_HEIGHT
        )
    )

    print(
        f"\nVideo resolution: "
        f"{width}x{height}"
    )

    print(
        f"Video FPS: {fps:.2f}"
    )

    print(
        "\nPress Q or ESC to stop."
    )

    print("=" * 65)

    # --------------------------------------------------------
    # Runtime variables
    # --------------------------------------------------------

    frame_count = 0

    start_time = time.time()

    # --------------------------------------------------------
    # Main loop
    # --------------------------------------------------------

    while True:

        ret, frame = cap.read()

        if not ret:
            break

        frame_count += 1

        # ====================================================
        # POSE EXTRACTION
        # ====================================================

        (
            kpts,
            bbox_diagonal,
            pose_confidence
        ) = extract_keypoints(
            pose_model,
            frame
        )

        # ====================================================
        # TEMPORAL CLASSIFICATION
        # ====================================================

        state = tracker.update(
            kpts,
            bbox_diagonal
        )

        # ====================================================
        # DRAW KEYPOINTS
        # ====================================================

        frame = draw_keypoints(
            frame,
            kpts
        )

        # ====================================================
        # DRAW INFORMATION
        # ====================================================

        frame = draw_info(
            frame,
            state,
            pose_confidence
        )

        # ====================================================
        # DISPLAY
        # ====================================================

        cv2.imshow(
            WINDOW_NAME,
            frame
        )

        # ====================================================
        # CONSOLE LOG
        # ====================================================

        if frame_count % 30 == 0:

            motion = state["motion"]
            posture = state["posture"]

            print(
                f"Frame {frame_count:5d} | "
                f"Motion={motion['label']:8s} "
                f"Score={motion['score']:.4f} | "
                f"Posture={posture['label']:8s} | "
                f"TailLow={state['tail']['low']} | "
                f"YOLO={pose_confidence:.2f}"
            )

        # ====================================================
        # KEYBOARD
        # ====================================================

        key = cv2.waitKey(1) & 0xFF

        if key == 27 or key == ord("q"):
            break

    # ========================================================
    # CLEANUP
    # ========================================================

    cap.release()

    cv2.destroyAllWindows()

    elapsed = time.time() - start_time

    print("\n")
    print("=" * 65)
    print("TEST COMPLETE")
    print("=" * 65)

    print(
        f"Frames processed : {frame_count}"
    )

    print(
        f"Processing time   : {elapsed:.2f} sec"
    )

    if elapsed > 0:

        print(
            f"Processing FPS    : "
            f"{frame_count / elapsed:.2f}"
        )

    print("=" * 65)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    main()