import cv2
import dlib
import pywt
import numpy as np
import json
import os
import firebase_admin
from firebase_admin import credentials,db 
import datetime
import time
from imutils import face_utils
from pynput import mouse, keyboard

# Function to return the face region
def cropped_image(img):
    gray_img=cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    faces = detector(gray_img)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        roi_color = img[y:y+h, x:x+w]
        return roi_color

# Function to apply Wavelet transform on the image
def w2d(img, mode='haar', level=1):
    imArray = img
    imArray = cv2.cvtColor(imArray,cv2.COLOR_BGR2GRAY)
    imArray =  np.float32(imArray)   
    imArray /= 255
    coeffs=pywt.wavedec2(imArray, mode, level=level)
    coeffs_H=list(coeffs)  
    coeffs_H[0] *= 0
    imArray_H=pywt.waverec2(coeffs_H, mode)
    imArray_H *= 255
    imArray_H =  np.uint8(imArray_H)
    return imArray_H

# Function to update the total work hour in the Firebase database
def update_workhour(starttime, endtime):
    emp_info = db.reference(f'Employees/{emp_id}').get()
    ref = db.reference(f'Employees/{emp_id}')
    workhour = emp_info['Total_WorkHour']
    worktime = int(workhour.split(":")[0]) * 3600 + int(workhour.split(":")[1]) * 60 + int(workhour.split(":")[2])
    active_time = (endtime - starttime) + worktime
    hours, remainder = divmod(active_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    emp_info['Total_WorkHour'] = f"{int(hours)}:{int(minutes)}:{int(seconds)}"
    ref.child('Total_WorkHour').set(emp_info['Total_WorkHour'])

# Function to stop the detection and update work hour after pressing pause buton
def stop_detection():
    global status,pause_flag
    pause_flag=0
    stoptime = time.time()
    emp_info = db.reference(f'Employees/{emp_id}').get()
    workhour = emp_info['Total_WorkHour']
    worktime = int(workhour.split(":")[0]) * 3600 + int(workhour.split(":")[1]) * 60 + int(workhour.split(":")[2])
    active_time = (stoptime - start_time) + worktime
    hours, remainder = divmod(active_time, 3600)
    minutes, seconds = divmod(remainder, 60)
    ref = db.reference(f'Employees/{emp_id}')
    emp_info['Total_WorkHour'] = f"{int(hours)}:{int(minutes)}:{int(seconds)}"
    ref.child('Total_WorkHour').set(emp_info['Total_WorkHour'])
    status=" Start: 's' & Quit: 'q'"

 # Function to detect work hours based on user activity after pressing start button  
def workhour_detection():
        global start_time,status,pause_flag,close_flag,last_activity_time
        last_activity_time = time.time()
        pause_flag=1
        br=0
        start_time = time.time()
        status=" Pause: 'p' & Quit: 'q'"
        predictor_path = "shape_predictor_68_face_landmarks.dat"
        predictor = dlib.shape_predictor(predictor_path)

        # function to Update last activity time based on mouse and keyboard events
        def on_move(x, y):
            global last_activity_time
            last_activity_time = time.time()
        
        def on_click(x, y, button, pressed):
            global last_activity_time
            last_activity_time = time.time()

        def on_scroll(x, y, dx, dy):
            global last_activity_time
            last_activity_time = time.time()

        def on_press(key):
            global last_activity_time
            last_activity_time = time.time()

        def on_release(key):
            global last_activity_time
            last_activity_time = time.time()

        # Start listeners for tracking mouse and keyboard events
        mouse_listener = mouse.Listener(on_move=on_move, on_click=on_click, on_scroll=on_scroll)
        mouse_listener.start()

        keyboard_listener = keyboard.Listener(on_press=on_press, on_release=on_release)
        keyboard_listener.start()
        
        # function to Calculate the Euclidean distance between two points ptA and ptB
        def compute(ptA, ptB):
            dist = np.linalg.norm(ptA - ptB)
            return dist
        
        # function to Determine if a person's eyes are active or inactive based on the ratio of distances between specific facial landmarks
        def blinked(a, b, c, d, e, f):
            up = compute(b, d) + compute(c, e)
            down = compute(a, f)
            ratio = up / (2.0 * down)
            if ratio > 0.25:
                return 1
            else:
                return 0

        first_time,third_time = None, None

        while True:
            _, frame1 = cap.read()
            gray = cv2.cvtColor(frame1, cv2.COLOR_BGR2GRAY)
            faces = detector(gray)
            for face in faces:
                x, y, w, h = face.left(), face.top(), face.width(), face.height()
                cv2.rectangle(frame1, (x, y), (x + w, y + h), col, 2)
            resized_frame1 = cv2.resize(frame1, (660, 415))
            cv2.rectangle(resized_frame1, (180,350), (480,390), (204,71,63), -1)
            cv2.putText(resized_frame1, text=" Pause: 'p' & Quit: 'q'", org=(187, 376), lineType=cv2.LINE_AA, fontScale=0.8, color=(255,255,255), thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
            template[94:94 + 415, 15:15 + 660] = resized_frame1
            cv2.imshow("Real-time Employee Attendance and Work Hour Tracking System", template)
            
            current_time = time.time()
            key = cv2.waitKey(1) & 0xFF
            if key == ord('p'):
                stop_detection()
                break
            elif key == ord('q'):
                close_flag=1
                on_closing()
                break
            elif current_time - last_activity_time >= 10:
                update_workhour(start_time, last_activity_time)
                status=" Start: 's' & Quit: 'q'"
                break

            elif len(faces) == 0:
                third_time = None
                if first_time is None:
                    first_time = current_time
                if current_time - first_time >= 16:
                    update_workhour(start_time, first_time)
                    status=" Start: 's' & Quit: 'q'"
                    break
            else:
                first_time = None
                for face in faces:
                    landmarks = predictor(gray, face)
                    landmarks = face_utils.shape_to_np(landmarks)
                    left_blink = blinked(landmarks[36], landmarks[37], landmarks[38], landmarks[41], landmarks[40], landmarks[39])
                    right_blink = blinked(landmarks[42], landmarks[43], landmarks[44], landmarks[47], landmarks[46], landmarks[45])
                    if left_blink == 0 or right_blink ==0:
                        if third_time is None:
                            third_time = current_time
                        elif current_time - third_time >= 3:
                            update_workhour(start_time, third_time)
                            status=" Start: 's' & Quit: 'q'"
                            br=1
                            break
                    else:
                        third_time = None
                if br==1:
                    break
        # stop listeners from tracking mouse and keyboard events
        mouse_listener.stop()
        keyboard_listener.stop()

# Function to recognise the face for registering attendance
def on_button_click():
    global status,no_face_start_time
    _, frame = cap.read()
    cropped_img=cropped_image(frame)
    registered_flag=1
    if cropped_img is None:
        no_face_start_time = time.time()
        cv2.putText(template, text="No Face Detected!!!", org=(380, 50), lineType=cv2.LINE_AA, fontScale=0.8, color=(255, 255, 255), thickness=2, fontFace=cv2.FONT_HERSHEY_SIMPLEX)
        status = " Retry: 'r' & Quit: 'q'"
        return
    
    scaled_raw_img=cv2.resize(cropped_img,(32,32))
    wt_img=w2d(cropped_img,'db1',5)
    scaled_wt_img=cv2.resize(wt_img,(32,32))
    full_img=np.vstack((scaled_raw_img.reshape(32*32*3,1),scaled_wt_img.reshape(32*32,1)))
    x=np.array(full_img).reshape(1,4096).astype(float)

    from joblib import dump,load
    model=load('classifier_model.joblib')
    y_predicted=model.predict(x)

    with open('celeb_dict.json',"r") as f:
        json_data=f.read()
        celeb_dict=json.loads(json_data)
        for key in celeb_dict.keys():
            if celeb_dict[key]==y_predicted:
                cv2.rectangle(template, (372,0), (674,93), (0,0,0), -1)
                folderpath="images"
                namelist=os.listdir(folderpath)
                for names in namelist:
                    if names==key:
                        global reg_flag
                        reg_flag=1
                        emp_img = os.listdir(os.path.join(folderpath,names))[0]
                        global emp_id
                        emp_id=emp_img.split(".")[0]
                        global col
                        col=(0, 255, 0)
                        emp_info=db.reference(f'Employees/{emp_id}').get()
                        cv2.putText(template, text="Name: "+emp_info['name'], org=(465, 44), lineType=cv2.LINE_AA, fontScale=0.6, color=(0, 255, 0), thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                        emp_img_display=cv2.imread(os.path.join(folderpath,names,emp_img))
                        cv2.putText(template, text="Emp ID: "+str(emp_id), org=(465, 24), lineType=cv2.LINE_AA, fontScale=0.7, color=(0, 255, 0), thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                        scaled_emp_img=cv2.resize(emp_img_display,(79,79))
                        template[8:8+79,380:380+79]=scaled_emp_img
                        x = datetime.datetime.now().strftime('%Y-%m-%d')
                        if(x!=emp_info['Last_attendance_date']):
                            ref=db.reference(f'Employees/{emp_id}')
                            emp_info['Total_attendance']+=1
                            ref.child('Total_attendance').set(emp_info['Total_attendance'])
                            emp_info['Last_attendance_date']=x
                            ref.child('Last_attendance_date').set(emp_info['Last_attendance_date'])
                            activetime=0
                            hours, remainder = divmod(activetime, 3600)
                            minutes, seconds = divmod(remainder, 60)
                            emp_info['Total_WorkHour']=f"{hours}:{minutes}:{seconds}"
                            ref.child('Total_WorkHour').set(emp_info['Total_WorkHour'])
                            registered_flag=0
                            cv2.putText(template, text="Registered", org=(465, 84), lineType=cv2.LINE_AA, fontScale=0.6, color=(0, 255, 0), thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                        if registered_flag==1:
                            cv2.putText(template, text="Already Registered", org=(465, 84), lineType=cv2.LINE_AA, fontScale=0.6, color=(0, 255, 0), thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                        cv2.putText(template, text="Attendance:"+str(emp_info['Total_attendance']), org=(465, 64), lineType=cv2.LINE_AA, fontScale=0.6, color=(0, 255, 0), thickness=1,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
                        status=" Start: 's' & Quit: 'q'"
                break

# function to perform necessary events when closing the window
def on_closing():
    if close_flag == 1:
        stoptime = time.time()
        update_workhour(start_time, stoptime)
    cap.release()
    cv2.destroyAllWindows()

cred = credentials.Certificate("employeedatarealtime-firebase-adminsdk-dw02o-9f8a7544ad.json")
firebase_admin.initialize_app(cred,{
     'databaseURL':"https://employeedatarealtime-default-rtdb.firebaseio.com/"
})

start_time=0
detector = dlib.get_frontal_face_detector()
cap = cv2.VideoCapture(0)
template=cv2.imread('template.jpg')
cap.set(3,320)
cap.set(4,240)
status="Register: 'r' & Quit: 'q'"
emp_id=None
reg_flag=0
pause_flag=0
close_flag=0
col=(204,71,63)
no_face_start_time = None

while True:
    _, frame = cap.read()
    resized_frame = cv2.resize(frame, (660, 415))
    gray = cv2.cvtColor(resized_frame, cv2.COLOR_BGR2GRAY)
    faces = detector(gray)
    for face in faces:
        x, y, w, h = face.left(), face.top(), face.width(), face.height()
        cv2.rectangle(resized_frame, (x, y), (x + w, y + h), col, 2)
    if no_face_start_time:
        elapsed_time = time.time() - no_face_start_time
        if elapsed_time >= 1:
            cv2.rectangle(template, (372, 0), (674, 93), (204, 71, 63), -1)
            no_face_start_time = None
            status = " Retry: 'r' & Quit: 'q'"
    cv2.rectangle(resized_frame, (180,350), (480,390), (204,71,63), -1)
    cv2.putText(resized_frame, text=status, org=(187, 376), lineType=cv2.LINE_AA, fontScale=0.8, color=(255,255,255), thickness=2,fontFace=cv2.FONT_HERSHEY_SIMPLEX)
    template[94:94 + 415, 15:15 + 660] = resized_frame
    cv2.imshow("Real-time Employee Attendance and Work Hour Tracking System",template)
    key = cv2.waitKey(1) & 0xFF
    if key == ord('r') and reg_flag==0:
        on_button_click()
    elif key == ord('s') and emp_id is not None:
        workhour_detection()
        if close_flag==1:
            break
    elif key == ord('p') and pause_flag==1:
        stop_detection()
    elif key == ord('q'):
        on_closing()
        break