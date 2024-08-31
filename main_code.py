import cv2
import numpy as np
from my_util import get_parking_spots_bboxes, empty_or_not

# comper the defrense between the state frame and the previes frame
def calc_diff(im1, im2):
    return np.abs(np.mean(im1) - np.mean(im2))

mask = "I:\\my_projects\\parking-space-counter\\RealTime-Parking-Space-Counter\\mask_1920_1080.png"
video_path= "I:\\my_projects\\parking-space-counter\\RealTime-Parking-Space-Counter\\parking_1920_1080_loop.mp4"

mask = cv2.imread(mask, 0)
cap = cv2.VideoCapture(video_path) 

connected_components = cv2.connectedComponentsWithStats(mask,4,cv2.CV_32S) # get the location of conected points of our park rectangle
spots = get_parking_spots_bboxes(connected_components) # make the classification predection 
spots_status = [None for j in spots]
diffs = [None for j in spots]

previous_frame = None

frame_num = 0
ret = True
step = 30
while ret:
    ret, frame = cap.read()
    
    # to calculate the defferint between frames that grater and apply the prediciton after 30 frame 
    if frame_num % step == 0 and previous_frame is not None:
        for spot_indx, spot in enumerate(spots):
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w]
            diffs[spot_indx] = calc_diff(spot_crop, previous_frame[y1:y1 + h, x1:x1 + w])

    if frame_num % step == 0:
        if previous_frame is None:
            arr = range(len(spots))
        else:
            [j for j in np.argsort(diffs) if diffs[j] / np.amax(diffs) > 0.4]
        for spot_indx in arr:
            spot = spots[spot_indx]
            x1, y1, w, h = spot
            spot_crop = frame[y1:y1 + h, x1:x1 + w]
            spot_status = empty_or_not(spot_crop)
            spots_status[spot_indx] = spot_status

    if frame_num % step == 0:
        previous_frame = frame.copy()
    # draw spots bbox 
    for spot_indx, spot in enumerate(spots): 
        spot_status = spots_status[spot_indx]
        x1,y1,w,h = spots[spot_indx]     
        if spot_status:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 255, 0), 2)
        else:
            frame = cv2.rectangle(frame, (x1, y1), (x1+w, y1+h), (0, 0, 255), 2)
    
    cv2.rectangle(frame, (80, 20), (525, 80), (0, 0, 0), -1)
    cv2.rectangle(frame, (90, 100), (200, 130), (0, 0, 0), -1), cv2.putText(frame, 'A1', (125, 125),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (90, 620), (210, 650), (0, 0, 0), -1), cv2.putText(frame, 'A2', (125, 645),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2) 
    cv2.rectangle(frame, (310, 100), (430, 130), (0, 0, 0), -1), cv2.putText(frame, 'B1', (350, 125),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    cv2.rectangle(frame, (320, 620), (440, 650), (0, 0, 0), -1), cv2.putText(frame, 'B2', (360, 642),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    cv2.rectangle(frame, (545, 60), (665, 90), (0, 0, 0), -1), cv2.putText(frame, 'C1', (585, 85),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    cv2.rectangle(frame, (545, 615), (665, 645), (0, 0, 0), -1), cv2.putText(frame, 'C2', (585, 642),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    cv2.rectangle(frame, (770, 130), (890, 160), (0, 0, 0), -1), cv2.putText(frame, 'D1', (810, 155),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)   
    cv2.rectangle(frame, (775, 615), (900, 645), (0, 0, 0), -1), cv2.putText(frame, 'D2', (815, 642),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
    cv2.rectangle(frame, (1000, 75), (1120, 105), (0, 0, 0), -1), cv2.putText(frame, 'E1', (1045, 100),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)  
    cv2.rectangle(frame, (1005, 532), (1125, 562), (0, 0, 0), -1), cv2.putText(frame, 'E2', (1045, 557),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (1230, 115), (1352, 145), (0, 0, 0), -1), cv2.putText(frame, 'F1', (1267, 140),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)    
    cv2.rectangle(frame, (1235, 530), (1355, 562), (0, 0, 0), -1), cv2.putText(frame, 'F2', (1275, 557),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (1460, 115), (1580, 145), (0, 0, 0), -1), cv2.putText(frame, 'G1', (1500, 140),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (1460, 527), (1580, 560), (0, 0, 0), -1), cv2.putText(frame, 'G2', (1500, 557),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (1690, 115), (1810, 145), (0, 0, 0), -1), cv2.putText(frame, 'H1', (1730, 140),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.rectangle(frame, (1690, 527), (1810, 560), (0, 0, 0), -1), cv2.putText(frame, 'H2', (1730, 557),cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.putText(frame,'Available Spots: {} / {}'.format(str(sum(spots_status)), str(len(spots_status))),(85,60), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2)
    cv2.namedWindow('frame', cv2.WINDOW_NORMAL)
    cv2.imshow('frame', frame)
    if cv2.waitKey(25) & 0xFF == ord('q'):
        break

    frame_num += 1

cap.release()
cv2.destroyAllWindows()
