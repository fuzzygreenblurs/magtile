import cv2
import numpy as np
import pdb
import time
import sys

# Calibration and conversion constants

MIN_X_PIX = 420
MAX_X_PIX = 1360
MIN_Y_PIX = 110
MAX_Y_PIX = 1030

SIZE_IN_PIX_X = MAX_X_PIX - MIN_X_PIX
SIZE_IN_PIX_Y = MAX_Y_PIX - MIN_X_PIX
SIZE_IN_INCHES = 11.875
PIX_TO_IN = SIZE_IN_INCHES / SIZE_IN_PIX_X

WORLD_ORIGIN_TRANSLATION = SIZE_IN_PIX_X / 2

# Color thresholds for red and yellow in RGB
lower_red = np.array([0, 0, 100])
upper_red = np.array([100, 100, 255])
lower_yellow = np.array([0, 100, 100])
upper_yellow = np.array([100, 255, 255])

background_frame = None
cap = cv2.VideoCapture(0)

# Minimum and maximum contour area thresholds
min_contour_area = 100
max_contour_area = 10000

while True:
    _, frame = cap.read()

    # pdb.set_trace()

    if frame is None:
        break

    # Crop the frame
    frame = frame[MIN_Y_PIX:MAX_Y_PIX, MIN_X_PIX:MAX_X_PIX]

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    gray = cv2.GaussianBlur(gray, (35, 35), 0)  # filter for noise between frames

    if background_frame is None:
        background_frame = gray

    frame_delta = cv2.absdiff(background_frame, gray)

    # Threshold the difference image
    _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
    thresh = cv2.dilate(thresh, None, iterations=6)

    # Detect red objects
    mask_red = cv2.inRange(frame, lower_red, upper_red)
    red_contours, _ = cv2.findContours(mask_red, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    # Detect yellow objects
    mask_yellow = cv2.inRange(frame, lower_yellow, upper_yellow)
    yellow_contours, _ = cv2.findContours(mask_yellow, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    red_pos = [0, 0]
    yellow_pos = [0, 0]
    # Process red contours
    for cnt in red_contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area or area > max_contour_area:
            continue  # Ignore small or large contours

        x, y, w, h = cv2.boundingRect(cnt)
        h_pos = ((x + w / 2) - WORLD_ORIGIN_TRANSLATION) * PIX_TO_IN
        v_pos = (-1 * ((y + h / 2) - WORLD_ORIGIN_TRANSLATION)) * PIX_TO_IN
        red_pos[0] = round(h_pos, 2) or 0
        red_pos[1] = round(v_pos, 2) or 0
        # red_pos = f"Red: ({round(h_pos, 2)}, {round(v_pos, 2)})"
        # print(f"Red: {round(h_pos, 2)}, {round(v_pos, 2)}")
        # sys.stdout.write(f"\rRed: {round(h_pos, 2)}, {round(v_pos, 2)}%")
        # sys.stdout.flush()

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
        cv2.putText(frame, "Red object detected", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

    # Process yellow contours
    for cnt in yellow_contours:
        area = cv2.contourArea(cnt)
        if area < min_contour_area or area > max_contour_area:
            continue  # Ignore small or large contours

        x, y, w, h = cv2.boundingRect(cnt)
        h_pos = ((x + w / 2) - WORLD_ORIGIN_TRANSLATION) * PIX_TO_IN
        v_pos = (-1 * ((y + h / 2) - WORLD_ORIGIN_TRANSLATION)) * PIX_TO_IN
        yellow_pos[0] = round(h_pos, 2) or 0
        yellow_pos[1] = round(v_pos, 2) or 0
        # print(f"Yellow: {round(h_pos, 2)}, {round(v_pos, 2)}")    
        # sys.stdout.write(f"\rYellow: {round(h_pos, 2)}, {round(v_pos, 2)}%")
        # yellow_pos = f"Yellow: ({round(h_pos, 2)}, {round(v_pos, 2)})"
        

        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
        cv2.putText(frame, "Yellow object detected", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

    sys.stdout.write(f"\r Red: ({red_pos[0]}, {red_pos[1]}), Yellow: ({yellow_pos[0]}, {yellow_pos[1]})")
    sys.stdout.flush()
    
    cv2.imshow('Frame', frame)
    # cv2.imshow('Gray', gray)
    # cv2.imshow('Threshold', thresh)
    # cv2.imshow('Red Mask', mask_red)
    # cv2.imshow('Yellow Mask', mask_yellow)

    k = cv2.waitKey(5)
    if k == 27:
        break

cv2.destroyAllWindows()
cap.release()