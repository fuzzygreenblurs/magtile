import cv2
import numpy as np
import pdb
import time
import csv

V_CALIBRATION  = 0
H_CALIBRATION  = 0

# # workspace box bounded to real world coordinates of: upper left:(-3, 3) to bottom right: (3, -3)
# PIX_RANGE      = 470
# MIN_X_PIX      = 825
# MAX_X_PIX      = MIN_X_PIX + PIX_RANGE
# MIN_Y_PIX      = 290
# MAX_Y_PIX      = MIN_Y_PIX + PIX_RANGE
# WORLD_ORIGIN_TRANSLATION = PIX_RANGE / 2

# SIZE_IN_INCHES = 5.9

# FLIR: workspace box bounded to real world coordinates of: upper left: (-0.3", 0.3") to bottom right: (4.4", 3.2")
PIX_RANGE_X    = 1024
PIX_RANGE_Y    = 1024
MIN_X_PIX      = 0
MAX_X_PIX      = MIN_X_PIX + PIX_RANGE_X
MIN_Y_PIX      = 0
MAX_Y_PIX      = MIN_Y_PIX + PIX_RANGE_Y
WORLD_ORIGIN_TRANSLATION = 0
WIDTH_IN_INCHES = 3.7
PIX_TO_IN      = WIDTH_IN_INCHES / PIX_RANGE_X

'''
camera field of view offsets:
1024x1024, x-offset: 700, y-offset: 226
'''

# SIZE_IN_INCHES = 5.9
# PIX_TO_IN      = SIZE_IN_INCHES / PIX_RANGE

frame_count = 0
background_frame = None
# cap = cv2.VideoCapture(1)
cap = cv2.VideoCapture('90fps-um.avi')

# Open a CSV file for writing
with open('um-positions.csv', mode='w', newline='') as file:
    writer = csv.writer(file)
    writer.writerow(["h_pos", "v_pos"])  # Write header

    start_time = time.time()
    while True:
        ret, frame = cap.read()
        frame_count += 1
        if not ret:
            break
        
        # Crop whole frame to target workspace area:
        # [Y, X] components respectively, given the current orientation of the camera
        workspace_frame = frame[MIN_Y_PIX : MAX_Y_PIX, MIN_X_PIX : MAX_X_PIX]

        gray = cv2.cvtColor(workspace_frame, cv2.COLOR_BGR2GRAY)
        gray = cv2.GaussianBlur(gray, (35, 35), 0)  # Filter for noise between frames

        if background_frame is None:
            background_frame = gray

        frame_delta = cv2.absdiff(background_frame, gray)

        # If any pixel intensity is < 25, set to 0. If > 25, set to 255
        # Average over 6 iterations
        _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY)
        thresh = cv2.dilate(thresh, None, iterations=6)

        # Note contours x,y positions are defined based on the cropped frame!
        contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        for cnt in contours:
            x, y, w, h = cv2.boundingRect(cnt)
            h_pos = ((x + w/2) - WORLD_ORIGIN_TRANSLATION) * PIX_TO_IN
            v_pos = (-1 * ((y + h/2) - WORLD_ORIGIN_TRANSLATION)) * PIX_TO_IN
            print(round(h_pos, 2), round(v_pos, 2))
            writer.writerow([round(h_pos, 2), round(v_pos, 2)])  # Write positions to CSV

        # cv2.imshow('gray', gray)
        cv2.imshow('color', workspace_frame)
        cv2.imshow('threshold', thresh)
        # print(frame.shape)  # 1080 * 1920 * 3 (rgb) -> (255, 255, 255)

        cv2.imshow('frame', frame)
        k = cv2.waitKey(5)
        if k == 27:
            break

end_time = time.time()
print(frame_count)
print(end_time - start_time)
cv2.destroyAllWindows()



# # https://www.youtube.com/watch?v=BURNRHK_r9g&ab_channel=AISearch
# # https://gist.github.com/fuzzygreenblurs/5cc899a1e077a5b1d7d484e0e4bbe6c6

# import cv2
# import numpy as np
# import pdb
# import time

# V_CALIBRATION  = 0
# H_CALIBRATION  = 0

# # # workspace box bounded to real world coordinates of: upper left:(-3, 3) to bottom right: (3, -3)
# # PIX_RANGE      = 470
# # MIN_X_PIX      = 825
# # MAX_X_PIX      = MIN_X_PIX + PIX_RANGE
# # MIN_Y_PIX      = 290
# # MAX_Y_PIX      = MIN_Y_PIX + PIX_RANGE
# # WORLD_ORIGIN_TRANSLATION = PIX_RANGE / 2

# # SIZE_IN_INCHES = 5.9

# # FLIR: workspace box bounded to real world coordinates of: upper left: (-0.3", 0.3") to bottom right: (4.4", 3.2")
# PIX_RANGE_X    = 1024
# PIX_RANGE_Y    = 1024
# MIN_X_PIX      = 0
# MAX_X_PIX      = MIN_X_PIX + PIX_RANGE_X
# MIN_Y_PIX      = 0
# MAX_Y_PIX      = MIN_Y_PIX + PIX_RANGE_Y
# WORLD_ORIGIN_TRANSLATION = 0
# WIDTH_IN_INCHES = 3.7
# PIX_TO_IN      = WIDTH_IN_INCHES / PIX_RANGE_X

# '''
# camera field of view offsets:
# 1024x1024, x-offset: 700, y-offset: 226
# '''

# # SIZE_IN_INCHES = 5.9
# # PIX_TO_IN      = SIZE_IN_INCHES / PIX_RANGE

# frame_count = 0
# background_frame = None
# # cap = cv2.VideoCapture(1)
# cap = cv2.VideoCapture('90fps-loop.avi')

# start_time = time.time()
# while True:
#     _, frame = cap.read()
#     frame_count += 1
#     if frame is None:
#         break
    
#     # crop whole frame to target workspace area:
#     # [Y, X] components respectively, given the current orientation of the camera
#     workspace_frame = frame[
#         MIN_Y_PIX : MAX_Y_PIX,
#         MIN_X_PIX : MAX_X_PIX
#     ]

#     gray = cv2.cvtColor(workspace_frame, cv2.COLOR_BGR2GRAY)
#     gray = cv2.GaussianBlur(gray, (35, 35), 0)                  # filter for noise between frames

#     if background_frame is None:
#         background_frame = gray

#     frame_delta = cv2.absdiff(background_frame, gray)

#     # if any pixel intensity is < 25, set to 0. if > 25, set to 255
#     # average over 6 iterations
#     _, thresh = cv2.threshold(frame_delta, 25, 255, cv2.THRESH_BINARY) 
#     thresh = cv2.dilate(thresh, None, iterations=6)

#     # note contours x,y positions are defined based on the cropped frame!
#     contours, heirarchy = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
#     for cnt in contours:
#         x, y, w, h = cv2.boundingRect(cnt)
#         h_pos = ((x + w/2) - WORLD_ORIGIN_TRANSLATION) * PIX_TO_IN
#         v_pos = (-1 * ((y + h/2) - WORLD_ORIGIN_TRANSLATION)) * PIX_TO_IN
#         print(round(h_pos, 2), round(v_pos, 2))

#     # cv2.imshow('gray', gray)
#     cv2.imshow('color', workspace_frame)
#     cv2.imshow('threshold', thresh)
#     # print(frame.shape)          # 1080 * 1920 * 3 (rgb) -> (255, 255, 255)


#     cv2.imshow('frame', frame)
#     k = cv2.waitKey(5)
#     if k == 27:
#         break

# end_time = time.time()
# print(frame_count)
# print(end_time - start_time)
# cv2.destroyAllWindows()

# '''
# for tomorrow, can we try swapping the measured x and y values as v_pos and h_pos respectively and THEN
# performing the coordinate transform? this should reduce confusion and simplify the coordinate transformation matrix
# '''
