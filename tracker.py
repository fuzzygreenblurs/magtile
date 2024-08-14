import redis
import cv2
import json
import numpy as np
import socket
import sys
import time

# Socket
SERVER_ADDR = ('localhost', 65432)
REDIS_CLIENT = redis_client = redis.StrictRedis(host='localhost', port=6379, db=0)
CHANNEL = "positions"

r = redis.Redis(host='localhost', port=6379, db=0)
stream_name = 'stream_positions'

# Calibration and conversion constants
MIN_X_PIX = 400
MAX_X_PIX = 1340
MIN_Y_PIX = 160
MAX_Y_PIX = 1080

SIZE_IN_CM = 30.1625
SIZE_IN_PIX_X = MAX_X_PIX - MIN_X_PIX
SIZE_IN_PIX_Y = MAX_Y_PIX - MIN_Y_PIX
PIX_TO_CM_X = SIZE_IN_CM / SIZE_IN_PIX_X
PIX_TO_CM_Y = SIZE_IN_CM / SIZE_IN_PIX_X
WORLD_ORIGIN_TRANSLATION_X = SIZE_IN_PIX_X / 2
WORLD_ORIGIN_TRANSLATION_Y = SIZE_IN_PIX_Y / 2

# SIZE_IN_PIX_X = MAX_X_PIX - MIN_X_PIX
# SIZE_IN_CM = 30.1625
# PIX_TO_CM = SIZE_IN_CM / SIZE_IN_PIX_X
# WORLD_ORIGIN_TRANSLATION = SIZE_IN_PIX_X / 2

# Color thresholds for red and yellow in RGB
lower_red = np.array([0, 0, 0])
upper_red = np.array([10, 10, 10])

lower_yellow = np.array([0, 100, 100])
upper_yellow = np.array([100, 255, 255])

# Minimum and maximum contour area thresholds
min_contour_area = 100
max_contour_area = 10000

def publish_position():
    background_frame = None
    cap = cv2.VideoCapture(0)

    while True:
        _, frame = cap.read()

        if frame is None:
            break

        # Crop the frame
        frame = frame[MIN_Y_PIX:MAX_Y_PIX, MIN_X_PIX:MAX_X_PIX]

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (35, 35), 0)

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
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            h_pos = ((x + w / 2) - WORLD_ORIGIN_TRANSLATION_X) * PIX_TO_CM_X
            v_pos = (-1 * ((y + h / 2) - WORLD_ORIGIN_TRANSLATION_Y)) * PIX_TO_CM_Y
            red_pos = [round(h_pos, 2), round(v_pos, 2)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 0, 255), 2)
            cv2.putText(frame, "Red object detected", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 0, 255), 2)

        # Process yellow contours
        for cnt in yellow_contours:
            area = cv2.contourArea(cnt)
            if area < min_contour_area or area > max_contour_area:
                continue

            x, y, w, h = cv2.boundingRect(cnt)
            h_pos = ((x + w / 2) - WORLD_ORIGIN_TRANSLATION_X) * PIX_TO_CM_X
            v_pos = (-1 * ((y + h / 2) - WORLD_ORIGIN_TRANSLATION_Y)) * PIX_TO_CM_Y
            yellow_pos = [round(h_pos, 2), round(v_pos, 2)]

            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 255), 2)
            cv2.putText(frame, "Yellow object detected", (x, y - 3), cv2.FONT_HERSHEY_SIMPLEX, 0.5, (0, 255, 255), 2)

        payload = {
            "timestamp": str(time.time()),  # Store as a string
            "yellow": json.dumps(yellow_pos),  # Serialize complex data as a string
            "red": json.dumps(red_pos)  # Serialize complex data as a string
        }

        # Add the payload to the Redis Stream
        if payload:
            r.xadd("stream_positions", payload)

        # print and display latest position and frame for diagnostics
        sys.stdout.write(f"\r Yellow: ({yellow_pos[0]}, {yellow_pos[1]}), cm: ({to_in(yellow_pos[0])}, {to_in(yellow_pos[1])})")
        # sys.stdout.write(f"\r Red: ({red_pos[0]}, {red_pos[1]}) , Yellow: ({yellow_pos[0]}, {yellow_pos[1]}), cm: ({to_in(yellow_pos[0])}, {to_in(yellow_pos[1])})")
        sys.stdout.flush()
        cv2.imshow('Frame', frame)

        k = cv2.waitKey(5)
        if k == 27:
            break

    cv2.destroyAllWindows()
    cap.release()
    

def to_in(cm):
    return round(cm * 0.393701, 2)

if __name__ == "__main__":
    publish_position()