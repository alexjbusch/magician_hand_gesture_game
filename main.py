import cv2
import mediapipe as mp
import time
from enum import IntEnum
import numpy as np
import math

cooldown_timer = time.time()
cooldown = 3

line_snap_distance = 50


cap = cv2.VideoCapture(0)
mpHands = mp.solutions.hands
hands = mpHands.Hands()
mpDraw = mp.solutions.drawing_utils

points = None

cTime = 0
pTime = 0

lines = []

line_primed = False
current_line = None

class HandPoint(IntEnum):
    LEFT_HAND = 0
    RIGHT_HAND = 1

    PALM = 0
    PALM_THUMB = 1
    
    THUMB_TIP = 4
    THUMB_MIDDLE = 3
    THUMB_BASE = 2
    

    INDEX_TIP = 8
    INDEX_UNDERTIP = 7
    INDEX_MIDDLE = 6
    INDEX_KNUCKLE = 5

    MIDDLE_TIP = 12
    MIDDLE_UNDERTIP = 11
    MIDDLE_MIDDLE = 10 
    MIDDLE_KNUCKLE = 9

    RING_TIP = 16
    RING_UNDERTIP = 15
    RING_MIDDLE = 14
    RING_KNUCKLE = 13

    PINKY_TIP = 20
    PINKY_UNDERTIP = 19
    PINKY_MIDDLE = 18
    PINKY_KNUCKLE = 17

class Gesture:
    def __init__(self, subGestures):
        self.subGestures = subGestures
    def check(self):

        for subGesture in self.subGestures:
            try:
                point0 = np.array(get_HandPoint_pos(subGesture[0][0],subGesture[0][1]))
                point1 = np.array(get_HandPoint_pos(subGesture[1][0],subGesture[1][1]))
                
                # handle unsupported operand error if one of the points is missing
                if (None in point0) or (None in point1):
                    return False
                euclidian_dist = np.sqrt(np.sum((point0 - point1)**2))
                # returns false if any subGesture is outside the threshold
                if euclidian_dist > subGesture[2]:
                    return False
                
            # if there is only one hand detected, catch the value error
            except ValueError:
                return False
            
        # returns true if all subGestures are within threshold distance
        return True



class Line:
    def __init__(self,start,end):
        self.start = start
        self.end = end
    def draw(self):
        cv2.line(flip, self.start, self.end,(0,253,255),2)
        

def draw_line_between_landmarks( hand1, handPoint1, hand2, handPoint2):
        try:
            # get the screen coordinates of a landmark on a hand
            screen_coords1 = get_HandPoint_pos(hand1,handPoint1)
            screen_coords2 = get_HandPoint_pos(hand2,handPoint2)

            current_line = Line(screen_coords1,screen_coords2)
            # this prevents a line from being drawn to the top left corner
            # whenever a landmark is not on screen but its hand is
            if (screen_coords1 == None) or (screen_coords2 == None):
                return

            # draw the line between the two landmarks
            cv2.line(flip, screen_coords1, screen_coords2,(255,100,100),2)

        # this occurs if one of the passed hands is not on screen 
        except ValueError:
            # return so it doesn't draw the line instead of crashing
            return



def get_HandPoint_pos(hand, handPoint):
        if hand <= len(points)-1:
            x1 = points[hand].landmark[handPoint].x
            y1 = points[hand].landmark[handPoint].y
            image_rows, image_cols, _ = flip.shape
            screen_coords = mpDraw._normalized_to_pixel_coordinates(x1, y1,
                                                       image_cols, image_rows)
            return screen_coords
        else:
            raise ValueError
        
    

    


def dot(vA, vB):
    return vA[0]*vB[0]+vA[1]*vB[1]
def get_angle(lineA, lineB):
    # Get nicer vector form
    vA = [(lineA[0][0]-lineA[1][0]), (lineA[0][1]-lineA[1][1])]
    vB = [(lineB[0][0]-lineB[1][0]), (lineB[0][1]-lineB[1][1])]
    # Get dot prod
    dot_prod = dot(vA, vB)
    # Get magnitudes
    magA = dot(vA, vA)**0.5
    magB = dot(vB, vB)**0.5
    # Get cosine value
    cos_ = dot_prod/magA/magB
    # Get angle in radians and then convert to degrees
    angle = math.acos(dot_prod/magB/magA)
    # Basically doing angle <- angle mod 360
    ang_deg = math.degrees(angle)%360

    if ang_deg-180>=0:
        # As in if statement
        return 360 - ang_deg
    else: 
        return ang_deg

def num_connected_lines():
    connected_lines = []
    for line in lines:
        start_connected = False
        end_connected = False
        for other in lines:
            if line != other:
                if line.start.all() == other.start.all() or line.start.all() == other.end.all():
                    start_connected = True
                if line.end.all() == other.start.all() or line.start.all() == other.end.all():
                    end_connected = True
        if start_connected and end_connected:
            connected_lines.append(line)
    return len(connected_lines)


# this is where gestures are defined, by passing a list of subGestures
gesture = Gesture([
    ((HandPoint.LEFT_HAND, HandPoint.INDEX_TIP),(HandPoint.LEFT_HAND, HandPoint.THUMB_TIP), 70),
    ((HandPoint.RIGHT_HAND, HandPoint.INDEX_TIP),(HandPoint.RIGHT_HAND, HandPoint.THUMB_TIP), 70),
    ])

end_gesture = Gesture([
    ((HandPoint.LEFT_HAND, HandPoint.MIDDLE_TIP),(HandPoint.LEFT_HAND, HandPoint.RING_TIP), 30),
    ((HandPoint.RIGHT_HAND, HandPoint.MIDDLE_TIP),(HandPoint.RIGHT_HAND, HandPoint.RING_TIP), 30),
    ])


playing = True
while playing:
    success, img = cap.read()
    #imgRGB = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    flip = cv2.flip(img,1)
    results = hands.process(flip)
    points = results.multi_hand_landmarks
    
    
    if points:
        for handLms in points:     
                mpDraw.draw_landmarks(flip, handLms)

        if gesture.check():
            draw_line_between_landmarks(HandPoint.LEFT_HAND, HandPoint.INDEX_TIP,
                                        HandPoint.RIGHT_HAND, HandPoint.INDEX_TIP)

            if time.time() - cooldown_timer > cooldown:
                
                if end_gesture.check():
                    point0 = np.array(get_HandPoint_pos(HandPoint.LEFT_HAND, HandPoint.INDEX_TIP))
                    point1 = np.array(get_HandPoint_pos(HandPoint.RIGHT_HAND, HandPoint.INDEX_TIP))
                    
                    # TODO: figure out a way to make this more dynamic and not repeat myself so much
                    current_lowest = line_snap_distance
                    for line in lines:
                        euclidian_dist = np.sqrt(np.sum((point0 - line.start)**2)) 
                        if euclidian_dist < line_snap_distance and euclidian_dist < current_lowest:
                            point0 = line.start
                            current_lowest = euclidian_dist
                        euclidian_dist = np.sqrt(np.sum((point0 - line.end)**2)) 
                        if euclidian_dist < line_snap_distance and euclidian_dist < current_lowest:
                            point0 = line.end
                            current_lowest = euclidian_dist


                        euclidian_dist = np.sqrt(np.sum((point1 - line.start)**2)) 
                        if euclidian_dist < line_snap_distance and euclidian_dist < current_lowest:
                            point1 = line.start
                            current_lowest = euclidian_dist
                        euclidian_dist = np.sqrt(np.sum((point1 - line.end)**2)) 
                        if euclidian_dist < line_snap_distance and euclidian_dist < current_lowest:
                            point1 = line.end
                            current_lowest = euclidian_dist

                    lines.append(Line(point0,point1))
                    #print(num_connected_lines())
                    cooldown_timer = time.time()

                    
        else:
            pass
            # check for end gesture
                    
    for line in lines:
        line.draw()

    cTime = time.time()
    fps = 1/(cTime-pTime)
    pTime = cTime

    cv2.putText(flip,str(fps),(10,70),cv2.FONT_HERSHEY_PLAIN,3, (255,0,255),3)
    
    cv2.imshow("Image",flip)
    cv2.waitKey(1)
