import numpy as np
import cv2
import argparse as ap

font = cv2.FONT_HERSHEY_SIMPLEX
THRESHOLD = 0.0008
FEATURE_TRACK_NUMBER = 13
MAX_THRESHOLD = 10
MIN_THRESHOLD = 2
feature_params = dict( maxCorners = FEATURE_TRACK_NUMBER,
                       qualityLevel = 0.02,
                       minDistance = 10,
                       blockSize = 7 )

lk_params = dict( winSize  = (15,15),
                  maxLevel = 2,
                  criteria = (cv2.TERM_CRITERIA_EPS | cv2.TERM_CRITERIA_COUNT, 10, 0.03))


parser = ap.ArgumentParser()
group = parser.add_mutually_exclusive_group(required=True)
group.add_argument('-v', "--videoFile", help="Path to Video File")
args = vars(parser.parse_args())
if args["videoFile"]:
        source = args["videoFile"]

cap = cv2.VideoCapture(source)
ret, old_frame = cap.read()
mask = np.zeros_like(old_frame)
fourcc = cv2.VideoWriter_fourcc(*'mp4v')
height, width = old_frame.shape[:2]
out = cv2.VideoWriter('./output_videos/result.mp4', fourcc, 20.0, (width/2, height/2))

count = 0
frame_count = 0

while(cap.isOpened()):

        ret,frame = cap.read()
        if not ret:
            print "Cannot capture frame device | CODE TERMINATION :( "
            exit()

        old_gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        feature_bottom = cv2.goodFeaturesToTrack(old_gray[height/2+height/5:height/2+2*height/5, 0:width], mask = None, **feature_params)
        feature_top = cv2.goodFeaturesToTrack(old_gray[0:height/5, 0:width], mask = None, **feature_params)
        ret,frame_new = cap.read()
        frame_gray = cv2.cvtColor(frame_new, cv2.COLOR_BGR2GRAY)
        frame_gray_bottom = frame_gray[height/2+height/5:height/2+2*height/5, 0:width]
        frame_gray_top = frame_gray[0:height/5, 0:width]
    
        track_feature_bottom, st, err = cv2.calcOpticalFlowPyrLK(old_gray[height/2+height/5:height/2+2*height/5, 0:width], frame_gray_bottom, feature_bottom.astype(np.float32), None, **lk_params)
        track_feature_top, st1, err1 = cv2.calcOpticalFlowPyrLK(old_gray[0:height/5, 0:width], frame_gray_top, feature_top.astype(np.float32), None, **lk_params)
        
        if track_feature_bottom is not None and track_feature_top is not None:
            if count<MAX_THRESHOLD:
                good_new_bottom = track_feature_bottom[st==1]
                good_old_bottom = feature_bottom[st==1]
                good_new_top = track_feature_top[st1==1]
                good_old_top = feature_top[st1==1]
                good_old = np.concatenate((good_old_bottom, good_old_top), axis=0)
                good_new = np.concatenate((good_new_bottom, good_new_top), axis=0)
                
                total_points = len(good_new)
                num_points_moved = 0
                for i,(new,old) in enumerate(zip(good_new,good_old)):
                    a,b = new.ravel()
                    b=int(b+height/2+height/5)
                    c,d = old.ravel()
                    d=int(d+height/2+height/5)
                    mask = cv2.line(mask, (a,b),(c,d),(0,255,0), 2)
                    if abs(a-c)<THRESHOLD or abs(b-d)<THRESHOLD:
                        num_points_moved +=1
                    
                    if(i == total_points-1):
                        if(num_points_moved >= FEATURE_TRACK_NUMBER):
                            frame_count += 1

                        if(num_points_moved <3):
                            frame_count = MIN_THRESHOLD

                        if(frame_count < 0):
                            frame_count = 0

                   
                    if(frame_count >= MAX_THRESHOLD):

                        image = cv2.resize(frame_new,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
                        cv2.putText(image,'STOP',(int(width/4)+250, int(height/4)-150), font, 0.6,(0,0,255),2,cv2.LINE_AA)
                        cv2.imshow('frame_new',image)
                        out.write(image)
                        
                        if k == 27:
                            break

                    else:
                        
                        image = cv2.resize(frame_new,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
                        cv2.imshow('frame_new',image)
                        out.write(image)
                   
                    
                    frame = cv2.circle(frame,(a,b),5,(0,255,0),-1)

                img = cv2.add(frame,mask)
                image = cv2.resize(img,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
            
                count = count + 1
                
                k = cv2.waitKey(1) & 0xff
                if k == 27:
                    break
            else:
                count = 0
        else:

            feature_bottom = cv2.goodFeaturesToTrack(old_gray[height/2+height/5:height/2+2*height/5, 0:width], mask = None, **feature_params)
            feature_top = cv2.goodFeaturesToTrack(old_gray[0:height/5, 0:width], mask = None, **feature_params)
            image = cv2.resize(img,(width/2, height/2), interpolation = cv2.INTER_CUBIC)
            cv2.imshow('frame_new',image)
            out.write(image)
            k = cv2.waitKey(1) & 0xff
            if k == 27:
                break
            continue
        old_gray = frame_gray.copy()
        feature_bottom = good_new.reshape(-1,1,2)

cv2.destroyAllWindows()
cap.release()
out.release()
