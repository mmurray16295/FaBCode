from ultralytics import YOLO
import cv2
import numpy as np
import mss

model = YOLO('runs/detect/train12/weights/best.pt')
# Define the screen region to capture (left, top, width, height)
# monitor = {"top": 100, "left": 100, "width": 800, "height": 600}

with mss.mss() as sct:
    # Select which monitor to capture (1 or 2)
    capture_monitor = 1  # Change to 2 to capture monitor 2
    monitor = sct.monitors[capture_monitor]
    print(sct.monitors)

    # Determine which monitor to display the window on
    display_monitor = 2 if capture_monitor == 1 else 1
    display_info = sct.monitors[display_monitor]

    cv2.namedWindow('YOLO Screen Capture', cv2.WINDOW_NORMAL)
    # Move the window to the opposite monitor
    cv2.moveWindow('YOLO Screen Capture', display_info["left"] + 100, display_info["top"] + 100)

    while True:
        img = np.array(sct.grab(monitor))
        frame = cv2.cvtColor(img, cv2.COLOR_BGRA2BGR)
        results = model(frame)
        annotated_frame = results[0].plot()
        cv2.imshow('YOLO Screen Capture', annotated_frame)
        key = cv2.waitKey(1)
        # If 'q' is pressed or window is closed, exit
        if key & 0xFF == ord('q'):
            break
        # If window is closed (cv2.getWindowProperty returns < 0), exit
        if cv2.getWindowProperty('YOLO Screen Capture', cv2.WND_PROP_VISIBLE) < 1:
            break

    cv2.destroyAllWindows()