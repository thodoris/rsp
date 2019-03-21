"""
Rock, Scissor, Paper stimulus training module
=====================================================================


"""
import sys
import os

from time import time
from optparse import OptionParser
from glob import glob
from random import choice
from collections import deque

import cv2

import numpy as np
from pandas import DataFrame
from psychopy import  core, event 
from pylsl import StreamInfo, StreamOutlet

from webcamvideostream import WebcamVideoStream

# parameters
cap_region_x_begin=0.5  # start point/total width
cap_region_y_end=0.8  # start point/total height
cap_detection_crop_pixels=10
hand_width , hand_height = 300,460



def is_gesture(frame,gesture):
    # Fist detection.
    detected = cv2.CascadeClassifier('./gestures/{0}_v4.xml'.format(gesture)).detectMultiScale(frame, 1.2, 4)
    return len(detected) == 1

def capture_gesture(frame):
    # 
    if (is_gesture(frame,'fist')):
        return 'rock'
    elif (is_gesture(frame,'palm')):
        return 'paper'
    elif (is_gesture(frame,'vick') or is_gesture(frame,'five')):
        return 'scissors'
    else:
        return 'unknown'

def load_images():
#-----------------------------------------------------------------------------
#       Load and configure image (.png with alpha transparency)
#-----------------------------------------------------------------------------
    # Load the source image
    imgs = {}
    
    
    for i in range(1, 5):
        # Load our overlay image
        img = cv2.imread('./images/{0}_transparent_xsmall.png'.format(str(i)),-1)
     
        # Create the mask for the img
        orig_mask = img[:,:,3]
     
        # Create the inverted mask for the hand
        orig_mask_inv = cv2.bitwise_not(orig_mask)
       
        # Convert hand image to BGR
        # and save the original image size (used later when re-sizing the image)
        img = img[:,:,0:3]
        imgs[i] = (img,orig_mask,orig_mask_inv)
    return imgs

def present(duration=120, isTest=False):
    
   
    NUM_GESTURES_BEFORE_DECIDE = 10
    
    GESTURE_BUFFER_LENGTH = 30
    GESTURE_BUFFER_DETECTION_LIMIT = 0.4 

    detected_gestures = deque(['unknown'],GESTURE_BUFFER_LENGTH)
    detected_gestures_high_level =  deque([],NUM_GESTURES_BEFORE_DECIDE)
    gesture = None
    round_winner = None
    markernames = {'rock':1 , 'scissors':2 , 'paper':3} 
    # Set up trial parameters
    n_trials = 2010
    iti = 2
    soa = 0.8
    jitter = 0.2
    record_duration = np.float32(duration)
    ii = 0
    iii = 0
    
    # Create markers stream outlet
    if not isTest:
        stream_info = StreamInfo('Markers', 'Markers', 1, 0, 'int32', 'myuidw12345_rsp')
        outlet = StreamOutlet(stream_info)

    # Setup trial list
    computer_selection = np.random.random_integers(3, size=n_trials)
    trials = DataFrame(dict(computer_selection=computer_selection,timestamp=np.zeros(n_trials)))

    # Setup graphics
    imgs = load_images()
    img=mask=mask_inv=img_question=None
    
   
    
    img_question = imgs[4][0]
    img_question_mask = imgs[4][1]
    img_question_mask_inv = imgs[4][2]
    #imgs["my_key"][0]
    
    #setup ranges for timing
    R_show_selection = range (7,10)
    R_detect_player_move = range (5,7)
    R_show_computer_move = range(2,5)
    R_show_winner = range(1,2)
    
   
    camera = WebcamVideoStream().start()
    #start recording
    start = time()
    main_timer = core.CountdownTimer(10)
    event_timestamp = None
    
    while camera.started:
    #for ii, trial in trials.iterrows():
        # Intertrial interval
        #core.wait(iti + np.random.rand() * jitter)
 
        frame = camera.read()

        frame = cv2.bilateralFilter(frame, 5, 50, 100)  # smoothing filter
        frame = cv2.flip(frame, 1)  # flip the frame horizontally
        cv2.rectangle(frame, (int(cap_region_x_begin * frame.shape[1]), 0),
                     (frame.shape[1], int(cap_region_y_end * frame.shape[0])), (255, 0, 0), 2)
        cv2.rectangle(frame, (0, 0),
                     (int(cap_region_x_begin * frame.shape[1]), int(cap_region_y_end * frame.shape[0])), (25, 25, 255), 2)
        
        timer_value= main_timer.getTime()
        
        # Capture marker timestamp (with video stimulus for beginning)
        if event_timestamp is None:
            event_timestamp = time() 
        elif timer_value>9.5:
            frame = np.zeros(frame.shape, dtype=np.uint8)
            
        # # Capture marker timestamp (with audio stimulus for beginning)
        # if event_timestamp is None:
            # audio_sound.stop()
            # event_timestamp = time() 
            # audio_sound.play()
        
        if int(timer_value) in R_show_selection:
            if int(timer_value)<10:
                txt="Rock"
                cv2.putText(frame, txt, (20, 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
                
            if int(timer_value)<9:
                txt="Scissors"
                cv2.putText(frame, txt, (120, 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
                
            if int(timer_value)<8:
                txt="Paper"
                cv2.putText(frame, txt, (240, 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
            #cv2.putText(frame, txt, (30, 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
        elif int(timer_value) in R_detect_player_move:
            #first choose computer move
            if img is None:
                # Select move
                label = trials['computer_selection'].iloc[ii]
                ii+=1
                print(label)
                img = imgs[label][0]
                mask = imgs[label][1]
                mask_inv = imgs[label][2]
            else: 
                # and display question box image
                roi = frame[0:hand_height, 0:hand_width]
                roi_bg = cv2.bitwise_and(roi,roi,mask = img_question_mask_inv)
                roi_fg = cv2.bitwise_and(img_question,img_question,mask = img_question_mask)
                dst = cv2.add(roi_bg,roi_fg)
            
                frame[0:hand_height, 0:hand_width] = dst

            #then detect player move    
            if (gesture is None or len(detected_gestures_high_level)<NUM_GESTURES_BEFORE_DECIDE) and timer_value>5.25 :
                crop_img = frame[cap_detection_crop_pixels:int(cap_region_y_end * frame.shape[0])-cap_detection_crop_pixels, 
                        int(cap_region_x_begin * frame.shape[1])+cap_detection_crop_pixels:frame.shape[1]-cap_detection_crop_pixels]
                grayframe = cv2.cvtColor(crop_img, cv2.COLOR_BGR2GRAY)
                tmp_gesture = capture_gesture(grayframe)
                if tmp_gesture!='unknown': 
                    detected_gestures.append(tmp_gesture)
                    
                    if len(detected_gestures)>int(GESTURE_BUFFER_LENGTH/2): #==detected_gestures.maxlen-1:
                        cnt , gesture = max(map(lambda val: (detected_gestures.count(val), val), detected_gestures))
                        if iii==0 or iii==5 or iii==10:
                            detected_gestures_high_level.append(gesture if cnt>=int(GESTURE_BUFFER_LENGTH//2 * GESTURE_BUFFER_DETECTION_LIMIT) else 'unknown')
                            print(detected_gestures_high_level)
                            print(iii)
                        elif iii>15:
                            iii=0;
                        iii+=1
            #final decision
            gesture = max(map(lambda val: (detected_gestures_high_level.count(val), val), set(detected_gestures_high_level)))[1] if detected_gestures_high_level else 'detecting...'
            cv2.putText(frame, gesture, (int(cap_region_x_begin * frame.shape[1])+cap_detection_crop_pixels, 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 2)
            
            #if we detected the gesture with high confidentiality , skip this part by altering the timer
            if gesture in ('rock','scissors','paper') and detected_gestures_high_level.count(gesture) > int(detected_gestures_high_level.maxlen * GESTURE_BUFFER_DETECTION_LIMIT):
                main_timer.add(-1*(timer_value-5))
                
            
        elif int(timer_value) in R_show_computer_move:
            #send marker
            if (not isTest) and  (gesture in ('rock','scissors','paper')):
                outlet.push_sample([markernames[gesture]] , event_timestamp)
            
            # show chosen computer move
            roi = frame[0:hand_height, 0:hand_width]
            roi_bg = cv2.bitwise_and(roi,roi,mask = mask_inv)
            roi_fg = cv2.bitwise_and(img,img,mask = mask)
            dst = cv2.add(roi_bg,roi_fg)
            
            frame[0:hand_height, 0:hand_width] = dst
        elif int(timer_value) in R_show_winner:
            #show winner
            
            if round_winner == None:
                if gesture not in ('rock','scissors','paper'):
                    round_winner = 'Invalid Match' 
                else:
                    round_winner = 'Player' if (label==1 and gesture=='paper') or (label==2 and gesture=='rock') or (label==3 and gesture=='scissors') else 'Computer'
                    if (label==1 and gesture=='rock') or (label==2 and gesture=='scissors') or (label==3 and gesture=='paper'):
                         round_winner = 'Draw'
                
                result_msg_x_pos = int(cap_region_x_begin * frame.shape[1])+cap_detection_crop_pixels if round_winner=='Player' else 30
            
            if round_winner in ('Draw' , 'Invalid Match'):
                cv2.putText(frame, round_winner, (240 , 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 3)
            else:
                cv2.putText(frame, 'WINNER', (result_msg_x_pos , 470), cv2.FONT_HERSHEY_TRIPLEX, 0.8, (255, 255, 255), 3)
            
      
        elif timer_value <0: 
            #clear variables
            iii=0
            img=mask=mask_inv=None
            gesture = None
            round_winner=None
            event_timestamp = None
            detected_gestures_high_level.clear()
            detected_gestures.clear()
            detected_gestures.append('unknown')
            main_timer.reset()
               

        cv2.imshow('original', frame)

        # Send marker
        #timestamp = time()
        #outlet.push_sample([markernames[label]], timestamp)
        #mywin.flip()

        # offset
        #core.wait(soa)
        #mywin.flip()
        k = cv2.waitKey(10) & 0xFF
        if (k == 27) or (time() - start) > record_duration: 
            break
        #event.clearEvents()

    # Cleanup & exiting
    camera.stop()
    cv2.destroyAllWindows()
    


def main():
    parser = OptionParser()

    parser.add_option("-d", "--duration",
                      dest="duration", type='int', default=120,
                      help="duration of the recording in seconds.")
    parser.add_option("-t", "--test",
                      dest="test", action="store_true" , default=False,
                      help="test only without actual recording the brainwaves.")                  

    (options, args) = parser.parse_args()
    present(options.duration , options.test)


if __name__ == '__main__':
    main()
