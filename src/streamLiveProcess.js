import axios from 'axios'
import { useRef } from 'react';
import * as np from 'numjs'
import * as pd from 'node-pandas'


function convertScale(img, alpha, beta){

    // Add bias and gain to an image with saturation arithmetics. Unlike
    // cv2.convertScaleAbs, it does not take an absolute value, which would lead to
    // nonsensical results (e.g., a pixel at 44 with alpha = 3 and beta = -210
    // becomes 78 with OpenCV, when in fact it should become 0).

    let  new_img = img * alpha + beta
    new_img[new_img < 0] = 0
    new_img[new_img > 255] = 255
    // NEED IMPROVEMENT
    return new_img.astype(np.uint8)

}

// function to perform array slicing
let slicedArray = (array,sx,ex,sy,ey) =>{
    return array.slice(sx, ex + 1).map(i => i.slice(sy, ey + 1))
}


function automatic_brightness_and_contrast(image, clip_hist_percent=1){
    // # gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    let gray = image

    // Calculate grayscale histogram
    // hist = cv2.calcHist([gray],[0],None,[256],[0,256])
    let hist = cv2.calcHist([gray],[0],None,[255],[0,255])
    let hist_size = len(hist)

    // # Calculate cumulative distribution from the histogram
    let accumulator = []
    accumulator.append(float(hist[0]))

    for(let i in hist_size){
        accumulator.append(accumulator[index -1] + float(hist[index]))
    }
        

    // Locate points to clip
    let maximum = accumulator[-1]
    clip_hist_percent = clip_hist_percent * (maximum/100.0)
    clip_hist_percent  = clip_hist_percent/ 2.0

    // Locate left cut
    let minimum_gray = 0
    while(accumulator[minimum_gray] < clip_hist_percent){
        minimum_gray += 1
    }
    // while accumulator[minimum_gray] < clip_hist_percent:
    //     minimum_gray += 1

    // Locate right cut
    let maximum_gray = hist_size -1
    while(accumulator[maximum_gray] >= (maximum - clip_hist_percent)){
        maximum_gray -= 1
    }
    // while accumulator[maximum_gray] >= (maximum - clip_hist_percent):
    //     maximum_gray -= 1

    // Calculate alpha and beta values
    let alpha = 255 / (maximum_gray - minimum_gray)
    let beta = -minimum_gray * alpha

    // '''
    // # Calculate new histogram with desired range and show histogram 
    // new_hist = cv2.calcHist([gray],[0],None,[256],[minimum_gray,maximum_gray])
    // plt.plot(hist)
    // plt.plot(new_hist)
    // plt.xlim([0,256])
    // plt.show()
    // '''

    let auto_result = convertScale(image, alpha=alpha, beta=beta)
    return (auto_result, alpha, beta)

}


function iris_coordinates(eye,detector_iris){
        // # compute the euclidean distances between the two sets of
        // # vertical eye landmarks (x, y)-coordinates
        let detect_flag = 0
        let x = eye.shape[1]/2 
        let y = eye.shape[0]/2     
        
        // # resize_scale_x = 3*0.4685
        // # resize_scale_y = 3*0.635
        
        let resize_scale_x = 1
        let resize_scale_y = 1
        
        
        // # eye = cv2.bilateralFilter(eye,3,5,5)
        // # eye = cv2.GaussianBlur(eye, (3, 5), 0)
        // # eye = cv2.GaussianBlur(eye, (5, 3), 0)
        // eye = np.array(automatic_brightness_and_contrast(eye, 5)[0])
        
        // # eye = cv2.GaussianBlur(eye, (3, 3), 2)
        // # eye = cv2.GaussianBlur(eye, (7, 7), 0)
        
        
        
        // # eye = cv2.GaussianBlur(eye, (7, 7), 0)
        
        eye = cv2.resize(eye,(int(round(eye.shape[1]*resize_scale_x)) ,int(round(eye.shape[0]*resize_scale_y))))
        
        // # eye = cv2.pyrUp(eye, (int(round(eye.shape[1]*resize_scale)) ,int(round(eye.shape[0]*resize_scale))))
        // # eye = cv2.pyrUp(eye)
                
        dets_iris = detector_iris(eye)

        for(const [index, element] of dets_iris.entries()){
                            // # print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(k, d.left(), d.top(), d.right(), d.bottom()))    
            if ((d.right() > (eye.shape[1]/5)) && (d.left() > (eye.shape[1]/5)) && (d.top() > 0) && (d.bottom() > 0)  && (np.abs(d.left() - d.right()) >= (eye.shape[1]/5))){
                x = int(round((d.left() + d.right())/(2*resize_scale_x)))
                y = int(round((d.top() + d.bottom())/(2*resize_scale_y)))
                detect_flag = 1
                // # print(detect_flag)
                // # x = (d.left() + d.right())//(2)
                // # y = (d.top() + d.bottom())//(2)
                cv2.rectangle(eye, (d.left(), d.top()), (d.right(), d.bottom()), (0, 0, 255), 5)
            } 
        }
                        
            // # if (np.abs(d.top() - d.bottom()) >= (eye.shape[1]/5)):                    
            
        let resize_scale = 1    
        cv2.imshow('eye',cv2.resize(eye,(int(round(eye.shape[1]*resize_scale)) ,int(round(eye.shape[0]*resize_scale)))))
        // # return the eye aspect ratio
        return [x, y, detect_flag]
}

function eye_aspect_ratio_left(eye){
    // # compute the euclidean distances between the two sets of
    // # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[0], eye[3])
    B = dist.euclidean(eye[4], eye[10])

    // # compute the euclidean distance between the horizontal
    // # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[5], eye[15])

    // # compute the eye aspect ratio
    ear = (A + B) / (2.5 * C)

    // # return the eye aspect ratio
    return ear
}

function eye_aspect_ratio_right(eye){
    // # compute the euclidean distances between the two sets of
    // # vertical eye landmarks (x, y)-coordinates
    A = dist.euclidean(eye[3], eye[4])
    B = dist.euclidean(eye[6], eye[14])

    // # compute the euclidean distance between the horizontal
    // # eye landmark (x, y)-coordinates
    C = dist.euclidean(eye[5], eye[1])

    // # compute the eye aspect ratio
    ear = (A + B) / (2.5 * C)

    // # return the eye aspect ratio
    return ear
}

function nose_pupil_dist_x(nose,leftEye,rightEye,left_pupil,right_pupil){
    left_pupil_dist_x1 = dist.euclidean(leftEye[5].T[0],(leftEye[5].T[0]-left_pupil[0]))
    left_pupil_dist_x2 = dist.euclidean(leftEye[5].T[0],(leftEye[15].T[0]+left_pupil[0]))
    right_pupil_dist_x1 = dist.euclidean(rightEye[5].T[0],(rightEye[5].T[0]-right_pupil[0]))
    right_pupil_dist_x2 = dist.euclidean(rightEye[5].T[0],(rightEye[1].T[0]+right_pupil[0]))
    // # left_pupil_dist = dist.euclidean(nose[4],(leftEye[3] + left_pupil))
    // # right_pupil_dist = dist.euclidean(nose[8],(rightEye[0] + right_pupil))    
    return (((left_pupil_dist_x2 - left_pupil_dist_x1) + (right_pupil_dist_x2-right_pupil_dist_x1))/2)
}

function nose_pupil_dist_y(nose,leftEye,rightEye,left_pupil,right_pupil){
    left_pupil_dist_y = dist.euclidean(nose.T[1],((leftEye[0].T[1]+leftEye[4].T[1])/2-left_pupil[1]))
    right_pupil_dist_y = dist.euclidean(nose.T[1],((rightEye[3].T[1]+rightEye[6].T[1])/2-right_pupil[1]))
    
    // # left_pupil_dist = dist.euclidean(nose[4],(leftEye[3] + left_pupil))
    // # right_pupil_dist = dist.euclidean(nose[8],(rightEye[0] + right_pupil))    
    return (left_pupil_dist_y-right_pupil_dist_y)
}

function left_right_eye_dist(leftEye,rightEye){
    let eye_dist = dist.euclidean(rightEye[3],leftEye[0])    
    return (eye_dist)
}

function eye_pupil_dist_x(leftEye,rightEye,left_pupil,right_pupil){
    let ep_dist_xs1 = dist.euclidean((rightEye[0].T[0]+right_pupil[0]),(rightEye[0].T[0]))
    let ep_dist_xs2 = dist.euclidean((leftEye[0].T[0]+left_pupil[0]),(leftEye[0].T[0]))
    let ep_dist_xe1 = dist.euclidean((rightEye[3].T[0]-right_pupil[0]),(rightEye[3].T[0]))
    let ep_dist_xe2 = dist.euclidean((leftEye[3].T[0]-left_pupil[0]),(leftEye[3].T[0]))
 
    let ep_dist_x = (ep_dist_xe1 + ep_dist_xe2 + ep_dist_xs1 + ep_dist_xs2)//4 
    
    
    
    
    return (ep_dist_x)
}


function eye_pupil_dist_y(leftEye,rightEye,left_pupil,right_pupil){
    let ep_dist_ys1 = dist.euclidean((np.min(rightEye.T[1])+right_pupil[1]),(np.min(rightEye.T[1])))
    let ep_dist_ys2 = dist.euclidean((np.min(leftEye.T[1])+left_pupil[1]),(np.min(leftEye.T[1])))
    let ep_dist_ye1 = dist.euclidean((np.max(rightEye.T[1])+right_pupil[1]),(np.max(rightEye.T[1])))
    let ep_dist_ye2 = dist.euclidean((np.max(leftEye.T[1])+left_pupil[1]),(np.max(leftEye.T[1])))
    
    let ep_dist_y = (ep_dist_ye1 + ep_dist_ye2 + ep_dist_ys1 + ep_dist_ys2)//4
    
    return (ep_dist_y)
}

function nose_jaw_dist(nose,jaw){
    if (nose.T[0] > jaw[26][0])
        right_extreme = dist.euclidean(nose.T[0],jaw[26][0])
    else right_extreme = 0
    if(nose.T[0] < jaw[17][0]) left_extreme = dist.euclidean(nose.T[0],jaw[17][0])
    else left_extreme = 0
    return (left_extreme - right_extreme)
}

function nose_jaw_dist_y(nose,jaw){
    if (nose.T[1]>(jaw[11][1]))
        top_extreme = dist.euclidean(nose.T[1],(jaw[11][1]))
    else
        top_extreme = 0
    if (nose.T[1]<(jaw[6][1]))
        bottom_extreme = dist.euclidean(nose.T[1],(jaw[6][1]))
    else
        bottom_extreme = 0
    return (bottom_extreme-top_extreme)
}

function nose_leftEye_dist_x(noseeye){
    nle_x = dist.euclidean(noseeye[0][0],noseeye[1][0])    
    return nle_x
}

function nose_rightEye_dist_x(noseeye){
    nre_x = dist.euclidean(noseeye[0][0],noseeye[2][0])    
    return nre_x
}

function left_rightEye_dist_x(noseeye){
    lre_x = dist.euclidean(noseeye[1][0],noseeye[2][0])    
    return lre_x
}

function nose_leftEye_dist_y(noseeye){
    nle_y = dist.euclidean(noseeye[0][1],noseeye[1][1])    
    return nle_y
}

function nose_rightEye_dist_y(noseeye){
    nre_y = dist.euclidean(noseeye[0][1],noseeye[2][1])    
    return nre_y    
}

function left_rightEye_dist_y(noseeye){
    lre_y = dist.euclidean(noseeye[1][1],noseeye[2][1])    
    return lre_y
}

function lips_aspect_ratio(mouth){
            // # compute the euclidean distances between the two sets of
            // # vertical eye landmarks (x, y)-coordinates
            A = dist.euclidean(mouth[4], mouth[18])
            B = dist.euclidean(mouth[5], mouth[16])

            // # compute the /euclidean distance between the horizontal
            // # eye landmark (x, y)-coordinates
            C = dist.euclidean(mouth[27], mouth[39])

            // # compute the eye aspect ratio
            mar = (A + B) / (2 * C)
                                        

            // # return the eye aspect ratio
            return mar
}

function mediapipe_face_landmarks(image, mp_drawing, mp_face_mesh, face_mesh, mp_face_detection, face_detection,face_detected){
    // # To improve performance, optionally mark the image as not writeable to
    // # pass by reference.
    let left_iris_coords = "",
    right_iris_coords = "",
    left_eye_coords = "",
    right_eye_coords = "",
    face_oval_coords = "",
    lips_coords = ""

    image.flags.writeable = false

    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    let results = face_mesh.process(image)
  
    // # Draw the face mesh annotations on the image.
    image.flags.writeable = true

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    landmark_coordinates = {}
    if(results.multi_face_landmarks){
            for(face_landmarks in results.multi_face_landmarks){
                landmark_coordinates = mp_drawing.landmarks_xy_idx(
                    image=image,
                    landmark_list=face_landmarks)
            }
    }
    // if results.multi_face_landmarks:
    //   for face_landmarks in results.multi_face_landmarks:    
    //    landmark_coordinates = mp_drawing.landmarks_xy_idx(
    //         image=image,
    //         landmark_list=face_landmarks)
       
       
    image.flags.writeable = False
    image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
    results = face_detection.process(image)
    
    // # detection_coordinates = {}
    // # if results.detections:
    // #   for detection in results.detections:    
    // #       detection_coordinates = mp_drawing.detection_xy_idx(image,detection)
    
    image.flags.writeable = True
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)
    
    let landmark_coordinates_array = [],
    landmark_coordinates_array_idx = [],
    landmark_coordinates_array_temp = [],
    idx = 0
    // SEE THIS TRY CATCH BLOCK

    try{

        
        if ((landmark_coordinates.length == 478)){
            face_detected = 1
            
            for (let value in range(landmark_coordinates.length)){
                landmark_coordinates_array_temp = np.array(landmark_coordinates[value])
                landmark_coordinates_array.append(landmark_coordinates_array_temp)
                idx = idx + 1
                landmark_coordinates_array_idx.append(idx)  
            }    
                          
            
            landmark_coordinates_array_idx = np.array(landmark_coordinates_array_idx)    
            landmark_coordinates_array = np.array(landmark_coordinates_array)   
            //   #   if len(np.array(landmark_coordinates_array_idx)) != 478:
            //   #       print('IDX is %d which is less than 478' % len(np.array(landmark_coordinates_array_idx)))
            
            left_iris_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS))))){

            }
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_LEFT_IRIS)))[value][0]
                left_iris_coords.append(landmark_coordinates_array[xy_temp])
            
            left_iris_coords = np.array([np.mean(np.array(left_iris_coords).T[0]),np.mean(np.array(left_iris_coords).T[1])]).astype(int)
            // # cv2.circle(image, left_iris_coords, 3, (0,0,255), -1)
            
            
            right_iris_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_RIGHT_IRIS))))){
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_RIGHT_IRIS)))[value][0]
                right_iris_coords.append(landmark_coordinates_array[xy_temp])
            }
                
          
            right_iris_coords = np.array([np.mean(np.array(right_iris_coords).T[0]),np.mean(np.array(right_iris_coords).T[1])]).astype(int)
            // # cv2.circle(image, right_iris_coords, 3, (0,0,255), -1)
            
            left_eye_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE))))){
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_LEFT_EYE)))[value][0]
                left_eye_coords.append(landmark_coordinates_array[xy_temp])
            }
                
                // # if value == 5 or value == 15 : # index 17 is the leftmost point on jaw line and index 26 is the rightmost point on jawline
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,255,255), -1)
                // # if value == 0  or value == 3: # index 6 is the lowest point on jaw line and index 11 is the highest point on face oval
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,0,255), -1)
                // # if value == 4  or value == 10: # index 6 is the lowest point on jaw line and index 11 is the highest point on face oval
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,0,255), -1)    
                
            left_eye_coords = np.array(left_eye_coords)    
            
            
            right_eye_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE))))){
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_RIGHT_EYE)))[value][0]
                right_eye_coords.append(landmark_coordinates_array[xy_temp])
            }
                // # if value == 1 or value == 15 : # index 17 is the leftmost point on jaw line and index 26 is the rightmost point on jawline
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,255,255), -1)
                // # if value == 3  or value == 4: # index 6 is the lowest point on jaw line and index 11 is the highest point on face oval
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,0,255), -1)
                // # if value == 6  or value == 14: # index 6 is the lowest point on jaw line and index 11 is the highest point on face oval
                // #     cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,0,255), -1)
           
            right_eye_coords = np.array(right_eye_coords)    
            
            lips_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_LIPS))))){
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_LIPS)))[value][0]
                lips_coords.append(landmark_coordinates_array[xy_temp])
            }
                
                // # cv2.circle(image, landmark_coordinates_array[xy_temp], 3, (255,255,255), -1)
              
            lips_coords = np.array(lips_coords)    
  
                  
  
            face_oval_coords = []
            for (let value in range(len(np.array(list(mp_face_mesh.FACEMESH_FACE_OVAL))))){
                xy_temp = (np.array(list(mp_face_mesh.FACEMESH_FACE_OVAL)))[value][0]
                face_oval_coords.append(landmark_coordinates_array[xy_temp])
            }
                
        
            face_oval_coords = np.array(face_oval_coords) 
        }
            

    }catch(err){

        print("Inside mediapipe function")
        print(e)

    }  
    return left_iris_coords, right_iris_coords, left_eye_coords, right_eye_coords, lips_coords, face_oval_coords, face_detected
  
}
// Till here

function count(s, c){
    // # Count variable
    res = 0
    
    for (i in range(len(s))){
        // # Checking character in string
        if (s[i] == c) res = res + 1
    }
        
        
    return res
}

// SEE default keyword once

function defaultFunction(o){
    if (isinstance(o, (datetime.date, datetime.datetime))) return o.isoformat()
}

function send_data(data){
    let user_id,ClassId,soc,ws
    cwd = os.path.abspath(os.path.dirname(__file__))
    try{

        // # soc.emit('data-analytics-shared', json.dumps(data,sort_keys=True,indent=1,default=default))
        data = json.dumps(data,defaultFunction)
        ws.send(data)

    }catch(err){
        console.log(`E-send-data(): ${err}`)
    }
}
      
function get_sec(time_str){
    // """Get Seconds from time."""
    h, m, s = time_str.split(':')
    return int(h) * 3600 + int(m) * 60 + int(s)
}






export const StreamLiveProcess = () => {
    console.log("Welcome to stream live process");

    try{

        let user_id, userName, id, accessToken, studyId, ClassId

        let sc_xr,norm_xr,classifier_xr,sc_yr,norm_yr,classifier_yr

        let mp_drawing,mp_face_mesh,face_mesh,mp_face_detection,face_detection

        let curtime, counter, ear_thresh,mar_thresh, mode, cam

        let screensizemm = [361, 203], screensize = [1366,768]

        videoRef = useRef(null);

        let zeros = (h, w, v = 0) => Array.from(new Array(h), _ => Array(w).fill(v));

        let m = 1

        console.log(zeros(1,2))

        let left_pupil_temp = zeros(1,2),right_pupil_temp=zeros(1,2)
        let left_pupil_mean = zeros(1,2), right_pupil_mean = zeros(1,2)

        let center_coordinates_x_temp = zeros(20,1), center_coordinates_y_temp = zeros(20,1)
        let center_coordinates_x_mean = 0 , center_coordinates_y_mean = 0
        center_coordinates_x_fn_max = -100000, center_coordinates_x_fn_min = 100000,
        center_coordinates_y_fn_max = -100000,

        center_coordinates_y_fn_min = 100000,

        center_coordinates_x_ep_max = -100000,
        center_coordinates_x_ep_min = 100000,


        center_coordinates_y_ep_max = -100000,
        center_coordinates_y_ep_min = 100000,

        nose = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
        leftEye = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
        rightEye = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],


        eye_counter = 0,

        left_pupil = [0,0],
        right_pupil = [0,0],

        left_pupil_old = [0,0],
        right_pupil_old = [0,0],

        njr = 0,
        njy = 0,
        ear = 0,
        mar = 0,

        left_eyes_center_x = 0,
        left_eyes_center_y = 0,

        right_eyes_center_x = 0,
        right_eyes_center_y = 0,
        jaw = [[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0],[0,0]],
        face_center_x = 0,
        face_center_y = 0,

        nose_center_x = 0,
        nose_center_y = 0,

        lred = 0,

        eyes_center_x = 0,
        eyes_center_y = 0,
        pupil_counter = 0,
        gaze_counter = 0,

        face_scale = 1,
        face_scale_mean = 1,

        frame_crop_x = 0,
        frame_crop_y = 0,

        face_scale_counter = 0,

        // for accessing the current directory but currently storing in localstorage
        // cwd = os.path.abspath(os.path.dirname(__file__))

        time_next = 0,
        detector_fer = FER(),
        FER_emotion = 'NA',
      
        text_filename = "predicted_output.txt",

        // Dealing with file system
        window.localStorage.setItem(text_filename,"LP,ES,DY,YF,HH,VH,GX,GY,XR,YR,FF,angry,disgust,fear,happy,sad,surprise,neutral,UE,CT,ST,ET,VT,FA")
        window.localStorage.setItem("ear_mar.txt","ear,mar,VT")


        // os.makedirs(os.path.dirname(os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename)), exist_ok=True)                
        // # text_file = open((os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename)), "a")
        // if os.path.exists((os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename))):
        //     text_file = open((os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename)), "a")
        // else:
        //     text_file = open(os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename), "w")
        //     text_file.write("LP,ES,DY,YF,HH,VH,GX,GY,XR,YR,FF,angry,disgust,fear,happy,sad,surprise,neutral,UE,CT,ST,ET,VT,FA\n")
        // text_file.close()

        // text_file = open(os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),"ear_mar.txt"), "w")
        // text_file.write("ear,mar,VT\n")
        // text_file.close()

        // text_file = open(os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),"logs.txt"), "w")
        // text_file.close()

        emotion = {'angry': 0.14,'disgust': 0.0,'fear': 0.01,'happy': 0.03,'sad': 0.08,'surprise': 0.0,'neutral': 0.74},

        shape = [[550, 576],
        [558, 622],[568, 666],[581, 708],[605, 743],[645, 765],[693, 775],[742, 781],[791, 778],[839, 777],[880, 767],[916, 749],[941, 718],[953, 678],[956, 633],[955, 588],[951, 542],[587, 543],[614, 523],[649, 516],[687, 516],[722, 525],[788, 517],[822, 505],[857, 500],[892, 503],[919, 519],[763, 561],[767, 589],[771, 617],[775, 646],[735, 670],[756, 674],[776, 677],[794, 671],[812, 667],[632, 573],[656, 560],[684, 557],[709, 572],[684, 578],[656, 580],[812, 566],[834, 547],[863, 546],[887, 556],[866, 566],[839, 569],[706, 718],[732, 705],[759, 697],[778, 701],[797, 695],[825, 702],[851, 712],[828, 726],[803, 733],[782, 735],[763, 735],[734, 729],[718, 718],[760, 715],[779, 716],[798, 713],[840, 713],[799, 709],
        [780, 711],[761, 709]],

        startTime = "",
        endTime = "",
        api_hit_one_time = 1,
        elapsed = 0,
        format = '%H:%M:%S',
        now = datetime.datetime.utcnow(),
        current_time = str(now.strftime("%H:%M:%S")),
        startDateTime = datetime.datetime.strptime(current_time, format),
        front_app = "FOCII.exe",
        videoTime_old = "0",
        previuos_UE = 0,
        previuos_UE_counter = 0


        while(mode != "END"){

            let face_present = 0,
            face_position_flag = 0  ,
            eyes_on_screen = 1,
            drowsy = 0,
            yawning = 0,
            Head_Horize = 0,
            Head_Vert = 0,
            user_eng = 0,
            ep_dist_x=0,ep_dist_y = 0,
            XR = screensize[0]/2,YR=screensize[1]/2,
            shape_flag = 0,
            videoTime = "0"

            // hit api call to get start and end time

            try{
                if(api_hit_one_time==1){
                    let url = "https://fociiapi.braina.live/studies/v1/get/study/scheduledclass?liveClassId=" + ClassId
                    axios.get(url).then(res =>{
                        if(res.status==200){
                            if(res.data == null) {
                                // continue
                                console.log(`NULL START END TIME`);
                            }else{
                                startTime = res.data.startTime.slice(11,19)
                                // startDateTime = datetime.datetime.strptime(startTime, format)
                                startDateTime = res.data.startDateTime
                                endTime = res.data.endTime
                                if(endtime != NULL){
                                    endTime = endTime.slice(11,19)
                                }
                                api_hit_one_time=0
                            }
                        }
                    })
                    .catch(err =>{
                        console.log(`Unable to get start and end time!!!`);
                        window.localStorage.setItem("GET Error",err);
                    })
                }

            }catch(err){
                window.localStorage.setItem("Error",err);
            }

            // next try catch
            try {
                
                let camframe = videoRef;

                if(camframe == null){
                    continue
                }

                camframe = np.array(automatic_brightness_and_contrast(camframe, 5)[0])

                camframe_fer = camframe
                
                camframe = cv2.resize(camframe, (int(screensize[0]*face_scale_mean),int(screensize[1]*face_scale_mean)))
                path = os.path.join(cwd,"brainalive","output","user_"+user_id,'video_'+ClassId,"ear_mar.txt")
                
                pupil_data = pd.read_csv(path)
                thresh_ear = pd.DataFrame()
                thresh_mar = pd.DataFrame()
                thresh_mar2 = pd.DataFrame()

                try{

                    thresh_ear['ear'] = pupil_data[pupil_data['ear']<0.30]['ear']
                    thresh_ear_max = thresh_ear['ear'].max()
                    if(thresh_ear_max) ear_thresh = thresh_ear_max
                    
                    thresh_mar['mar'] = pupil_data[(pupil_data['mar']<0.45)]['mar']
                    thresh_mar2['mar'] = thresh_mar[(thresh_mar['mar']>0.2)]['mar']
                    thresh_mar_mean = thresh_mar2['mar'].mean()
                    if(thresh_mar_mean) mar_thresh= thresh_mar_mean
                        // # print("ear_thresh: ",ear_thresh)
                        // # print("mar_thresh: ",mar_thresh)

                }catch(err){
                    console.log(`Pandas : ${err}`)
                }

                face_scale_old = face_scale_mean

                // another try catch

                try{

                    left_iris_coords, right_iris_coords, left_eye_coords, right_eye_coords, lips_coords, face_oval_coords, face_present = mediapipe_face_landmarks(camframe, mp_drawing, mp_face_mesh, face_mesh, mp_face_detection, face_detection, face_present)

                }   catch(err){
                    console.log(`E_medipipe: ${err}`)
                }    
                    
                    
                gray = cv2.cvtColor(camframe, cv2.COLOR_BGR2GRAY)



            } catch (error) {
                console.log(`E1:${error} `)
            }


            let radius = 30,
            radius_boundry = 36,
            radius_border = 32,
            colour = (0,0,255),
            colour_boundry = (0,0,0),
            thickness = -1,

            format = '%H:%M:%S',

            now = datetime.datetime.utcnow(),
            current_time = str(now.strftime("%H:%M:%S"))
            
            videoTime = "0"
            
            if(previuos_UE_counter<3) previuos_UE_counter+=1
            else previuos_UE_counter = 0
            
            // another try catch
            
            try{

                endDateTime = datetime.datetime.strptime(current_time, format)
                // #print("start: {} , curTime: {}".format(startDateTime,endDateTime))/

                if(startDateTime != null && (str(startDateTime) <= str(endDateTime)))
                    videoTime = get_sec(str(endDateTime - startDateTime))
                else videoTime = "0"

            }catch(err){
                console.log(`E2(ii): ${err}`)
            }

            // try catch to detect app running in frontend or not
            try{
                front_app = str(gw.getActiveWindow()).strip() + ".exe"
            }catch(err){
                console.log(`${err}`)
            }

            if(face_present){
                try {

                    face_center_x = int(((face_oval_coords[26][0] + face_oval_coords[17][0])/2))
                    face_center_y = int(((face_oval_coords[11][1] + face_oval_coords[6][1])/2))
                    face_width = int((abs(face_oval_coords[17][0] - face_oval_coords[26][0])))
                    face_height = int((abs(face_oval_coords[6][1] - face_oval_coords[11][1])))
                    // # print(face_center_x," ",face_center_y," ",face_width," ",face_height)
                    // # print(camframe.shape[1]//3," ",camframe.shape[1]*2//3," ",camframe.shape[0]//3," ",camframe.shape[0]*2//3," ",camframe.shape[1]//6," ",camframe.shape[0]//4)
                    // # screen shape ka 1/3 to 2/3 inside this range
                    if ((face_center_x > (camframe.shape[1]/3)) && (face_center_x < (camframe.shape[1]*2/3)) && (face_center_y > (camframe.shape[0]/3)) && (face_center_y < (camframe.shape[0]*2/3)) && (face_width >= camframe.shape[1]/6) && (face_height >= camframe.shape[0]/4)         )
                        face_position_flag = 1
                    // # print(face_position_flag)
                } catch (error) {
                    console.log(`Face position flag error : ${error}`)
                }
            }else if(int(videoTime)!=videoTime_old && videoTime!="0"){
                // #print("videoTime",videoTime)
                // #print("videotime_old",videoTime_old)
                now = datetime.datetime.utcnow()
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")
                try{

                    // text_file = open((os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename)), "a")
                    // print(str(int(elapsed + float(curtime))))
                    // text_file.write("%d,%d,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%s\n"%(0,0,0,0,0,0,0,0,screensize[0]/2,screensize[1]/2,face_position_flag,0,0,0,0,0,0,0,0,dt_string,startTime,endTime,videoTime,front_app))
                    // text_file.close()
                    data = {"data-analytics-shared": {
                            'x-access-token':accessToken,
                            'study-id':studyId,
                            'class-id':ClassId,
                            'data':{'userId':id,'UserName':userName,'LP':0,'ES':0,'DY':0,'YF':0,'HH':0,'VH':0,'GX':0,'GY':0,'XR':XR,'YR':YR,'FF':face_position_flag,'angry':0,'disgust':0,'fear':0,'happy':0,'sad':0,'surprise':0,'neutral':0,'UE':0,'CT':dt_string,'ST':startTime,'ET':endDateTime,'VT':videoTime,'FA':front_app,'PUE':previuos_UE}
                           }
                        }

                    console.log(`Data writing to zeroes: ${data}`);
                    let T = threading.Thread(target=send_data(data))
                    T.start()
                    // # filename = os.path.join(cwd,"brainalive","output","user_"+str(user_id),"video_"+str(ClassId), "images",str(videoTime) +".jpg")
                    // # os.makedirs(os.path.dirname(os.path.join(cwd, filename)), exist_ok=True) 
                    // # cv2.imwrite(filename, camframe,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                    videoTime_old = int(videoTime)
                    if(previuos_UE_counter==2) previuos_UE = 0
                    }catch(err){
                    console.log(`E3 : ${err}`)
                }
                end = time.time()
                elapsed = end - start
                continue
            }

            else{
                continue
            }



            // another try catch

            try {

                left_pupil = list(left_iris_coords)
            right_pupil = list(right_iris_coords)
            
            
            detect_flag_left = 0
            detect_flag_right = 0
            if(left_pupil!=right_pupil){
                if(left_pupil!=None)          
                // # left_pupil = (left_pupil[0], left_pupil[1])
                detect_flag_left = 1        
                if(left_pupil!=None)           
                    // # right_pupil = (right_pupil[0], right_pupil[1])
                detect_flag_right = 1
            }      
                
            
            face_center_x = int((face_oval_coords[26][0] + face_oval_coords[17][0])/2)
            face_center_y = int((face_oval_coords[11][1] + face_oval_coords[6][1])/2)
            face_width = int(abs(face_oval_coords[17][0] - face_oval_coords[26][0]))
            face_height = int(abs(face_oval_coords[6][1] - face_oval_coords[11][1]))

            rightEye = right_eye_coords
            leftEye = left_eye_coords
            // # nose = np.array(nose_tip_coordinates)
            nose = np.array([(np.array(face_oval_coords)[6][0] + np.array(face_oval_coords)[11][0])/2,(np.array(face_oval_coords)[17][1] + np.array(face_oval_coords)[26][1])/2 ])
            lips = np.array(lips_coords)
            jaw = np.array(face_oval_coords)
            nose_center_x = nose.T[0]
            nose_center_y = nose.T[1]
            both_eyes = np.concatenate((leftEye,rightEye), axis=0)
            // # predefined libarary FER emotions detect emotions
            FER_emotion = detector_fer.detect_emotions(camframe_fer)
            
            left_eyes_center_x = int((leftEye.T[0].min() + leftEye.T[0].max())/2)
            left_eyes_width = int(abs(leftEye.T[0].min() - leftEye.T[0].max()))            
            left_eyes_center_y = int((leftEye.T[1].min() + leftEye.T[1].max())/2)
            left_eyes_height = int(abs(leftEye.T[1].min() - leftEye.T[1].max()))
            
            right_eyes_center_x = int((rightEye.T[0].min() + rightEye.T[0].max())/2)
            right_eyes_width = int(abs(rightEye.T[0].min() - rightEye.T[0].max()))
            right_eyes_center_y = int((rightEye.T[1].min() + rightEye.T[1].max())/2)        
            right_eyes_height = int(abs(rightEye.T[1].min() - rightEye.T[1].max()))
            
            eyes_center_x = int((both_eyes.T[0].min() + both_eyes.T[0].max())/2)
            eyes_center_y = int((both_eyes.T[1].min() + both_eyes.T[1].max())/2)

            eyes_width = int(abs(both_eyes.T[0].min() - both_eyes.T[0].max()))
            eyes_height = int(abs(both_eyes.T[1].min() - both_eyes.T[1].max())) 
            
            
            let marginx = 5,
            marginy = 10 ,

            
            leftEyeImage_fp_sx  = Math.min(...leftEye[1]) -marginy,
            leftEyeImage_fp_ex = Math.max(...leftEye[1]) + marginy,
            leftEyeImage_fp_sy = Math.min(...leftEye[0]) - marginx,
            leftEyeImage_fp_ey = Math.max(...leftEye[0]) + marginx,

            rightEye_image_fp_sx  = Math.min(...rightEye[1]) -marginy,
            rightEye_image_fp_ex = Math.max(...rightEye[1]) + marginy,
            rightEye_image_fp_sy = Math.min(...rightEye[0]) - marginx,
            rightEye_image_fp_ey = Math.max(...rightEye[0]) + marginx,
        
            leftEye_image = slicedArray(gray,leftEyeImage_fp_sx,leftEyeImage_fp_ex,leftEyeImage_fp_sy,leftEyeImage_fp_ey),
            // leftEye_image = gray[leftEye.T[:][1].min()-marginy : leftEye.T[:][1].max()+marginy,leftEye.T[:][0].min()-marginx: leftEye.T[:][0].max()+marginx],
            
            rightEye_image = slicedArray(gray,rightEye_image_fp_sx,rightEye_image_fp_ex,rightEye_image_fp_sy,rightEye_image_fp_ey) 
        
            leftEAR = eye_aspect_ratio_left(leftEye),
            rightEAR = eye_aspect_ratio_right(rightEye),
            ear = (leftEAR + rightEAR)/2,
            mar = lips_aspect_ratio(lips_coords),
            njy = nose_jaw_dist_y(nose,jaw),
            
            njr = nose_jaw_dist(nose,jaw),
            lred = left_right_eye_dist(leftEye,rightEye)


                
            } catch (error) {
                console.log(`E5 ${err}`)
            }

            // another try catch

            try {

                if (detect_flag_left == 1 && detect_flag_right == 1){
                    left_pupil_pre = [0,0]
                    right_pupil_pre = [0,0]
                    left_pupil_pre[0] = left_pupil[0] - Math.min(...leftEye[0])
                    left_pupil_pre[1] = left_pupil[1] - Math.min(...leftEye[1])
                    left_pupil = left_pupil_pre
                    right_pupil_pre[0] = right_pupil[0] -  Math.min(...rightEye[0])
                    right_pupil_pre[1] = right_pupil[1] - Math.min(...rightEye[1])
                    right_pupil = right_pupil_pre
                }
                    
                
                
                if (detect_flag_left == 1 && detect_flag_right == 0){
                    right_pupil = [0,0]
                    right_pupil[0] = left_pupil[0] - Math.min(...leftEye[0])
                    right_pupil[1] = left_pupil[1] - Math.min(...leftEye[1])
                    
                }
                    
                    
                if (detect_flag_left == 0 && detect_flag_right == 1){
                    left_pupil = [0,0]
                    left_pupil[0] = right_pupil[0] - Math.min(...rightEye[0])
                    left_pupil[1] = right_pupil[1] -  Math.min(...rightEye[1])
                }
                
                if (detect_flag_left == 0 && detect_flag_right == 0){
                    left_pupil = left_pupil_old
                    right_pupil = right_pupil_old
                }    
                    
                
                left_pupil_old = [0,0]
                right_pupil_old = [0,0]
                left_pupil_old[0] = left_pupil[0]
                left_pupil_old[1] = left_pupil[1]
                right_pupil_old[0] = right_pupil[0]
                right_pupil_old[1] = right_pupil[1]
                


            } catch (error) {
                console.log(`E6 : ${error}`)
            }

            // another try catch

            try {

                left_pupil_temp[pupil_counter] =  [left_pupil[0],left_pupil[1]]
                right_pupil_temp[pupil_counter] = [right_pupil[0],right_pupil[1]]            
                pupil_counter += 1  
                
                if( pupil_counter == m)              
                    pupil_counter = 0        
                    
                left_pupil_mean = [int(round(np.mean(left_pupil_temp.T[0]))),int(round(np.mean(left_pupil_temp.T[1])))]
                right_pupil_mean = [int(round(np.mean(right_pupil_temp.T[0]))),int(round(np.mean(right_pupil_temp.T[1])))]

                ep_dist_x = nose_pupil_dist_x(nose,leftEye,rightEye,left_pupil_mean,right_pupil_mean)
                ep_dist_y = nose_pupil_dist_y(nose,leftEye,rightEye,left_pupil_mean,right_pupil_mean)
                
                noseeye = np.concatenate((np.reshape(nose,(-1, 2)), np.array([[left_eyes_center_x,left_eyes_center_y], [right_eyes_center_x,right_eyes_center_y]])), axis=0)
                
                nle_x = nose_leftEye_dist_x(noseeye)            
                nre_x = nose_rightEye_dist_x(noseeye)            
                lre_x = left_rightEye_dist_x(noseeye)            
                
                nle_y = nose_leftEye_dist_y(noseeye)            
                nre_y = nose_rightEye_dist_y(noseeye)            
                lre_y = left_rightEye_dist_y(noseeye)            

                tri_peri = nle_x + nre_x + lre_x + nle_y + nre_y + lre_y
                
                // #if (abs(ep_dist_x) > 6) or (abs(ep_dist_y) > 12) or (abs(njr) > 150) or (njy > 180) or (njy < -100) or (face_present==0) :        
                // #if (abs(ep_dist_x) > 30) or (abs(ep_dist_y) > 10) or (abs(njr) > 150) or (njy > 180) or (njy < -100) or (face_present==0) or (ear < ear_thresh*0.6):     

                // # if ((abs(njr) > 80*510//tri_peri) or (njy > 210*500//tri_peri) or (njy< -100*510//tri_peri) or (jaw_max > eye_max*1.02) or (jaw_max < eye_min*0.99)): 
                // #     eyes_on_screen = 0
                // # if ((abs(njr) > 80*480//tri_peri) or (njy > 210*470//tri_peri) or (njy < -100*480//tri_peri) or (jaw_max > eye_max*1.02) or (jaw_max < eye_min*0.99)):
                // # if ((abs(njr) > 80*480//tri_peri) or (njy > 210*670//tri_peri) or (njy < -100*670//tri_peri)) : 
                // #     eyes_on_screen = 0

                if (ear < ear_thresh*0.7)       
                    drowsy = 1
                
                if (mar > mar_thresh*1.1)
                    yawning = 1

                if (abs(njr) >= 180)
                    Head_Horize = 100
                else
                    Head_Horize = 100 * abs(njr)/180

                if ((abs(njy) > 60))
                    Head_Vert = 100
                else
                    Head_Vert = 100 * abs(njy)/60

                Head_Horize = Head_Horize*400/tri_peri
                Head_Vert = Head_Vert*400/tri_peri

                if(Head_Horize > 100)
                    Head_Horize = 100
                if(Head_Vert > 100)
                    Head_Vert = 100

                if((Head_Horize > 50))
                // # or (Head_Vert > 85)):
                    eyes_on_screen = 0
                
                if (face_present)
                    user_eng = ((1 - ((1-eyes_on_screen)*0.5 + drowsy*0.3 + yawning*0.2))*100)
                else
                    user_eng = 0.0
                
                
                        
                let absolute_center_coordinates_x = (screensize[0]/2),
                absolute_center_coordinates_y = (screensize[1]/2),
                
                
                absolute_center_diff_x = absolute_center_coordinates_x - (face_center_x),
                
                absolute_center_diff_y = absolute_center_coordinates_y - (face_center_y),
                
                    
                center_coordinates_x_e = (int(np.mean(np.concatenate((leftEye[0], rightEye[0]), axis = 0)))),
                center_coordinates_y_e = (int(np.mean(np.concatenate((leftEye[1], rightEye[1]), axis = 0))))
                
            } catch (error) {
                    console.log(`E8:${error}`)
            }

            // track of current time
            try {

                time_now = time.time()
                fps = round(1/(time_now-time_next))
                
                time_next = time.time()
                
                now = datetime.datetime.utcnow()  
                dt_string = now.strftime("%d/%m/%Y %H:%M:%S")

                if(len(FER_emotion)>0)
                    emotion = FER_emotion[0]['emotions']

                try {

                    if(int(videoTime)!=int(videoTime_old)){
                                // # print("videoTime",videoTime)
                        // # print("videotime_old",videoTime_old)
                        try {

                            text_file = open((os.path.join(cwd,"brainalive","output", "user_"+user_id,"video_"+ClassId,"ear_mar.txt")), "a")
                            if(ear && mar){
                                text_file.write("%.2f,%.2f,%s\n"%(ear,mar,videoTime))
                                text_file.close()
                            }
                               
                            
                        } catch (error) {
                            console.log(`E4: ${error}`)
                        }

                        print(os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename))
                        text_file = open((os.path.join(cwd,"brainalive","output", "user_"+str(user_id),"video_"+str(ClassId),text_filename)), "a")
                        print(str(int(elapsed + float(curtime))))
                        text_file.write("%d,%.2f,%d,%d,%d,%d,%d,%d,%.3f,%.3f,%d,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%.2f,%s,%s,%s,%s,%s\n"%(face_present,eyes_on_screen,drowsy,yawning,Head_Horize,Head_Vert,ep_dist_x,ep_dist_y,screensize[0]/2,screensize[1]/2,face_position_flag,emotion['angry'], emotion['disgust'],emotion['fear'],emotion['happy'],emotion['sad'],emotion['surprise'],emotion['neutral'],user_eng,dt_string,startTime,endTime,videoTime,front_app))
                        text_file.close()
                        data = {"data-analytics-shared": {
                                'x-access-token':accessToken,
                                'study-id':studyId,
                                'class-id':ClassId,
                                'data':{'userId':id,'UserName':userName,'LP':face_present,'ES':eyes_on_screen,'DY':drowsy,'YF':yawning,'HH':Head_Horize,'VH':Head_Vert,'GX':ep_dist_x,'GY':ep_dist_y,'XR':XR,'YR':YR,'FF':face_position_flag,'angry':emotion['angry'],'disgust':emotion['disgust'],'fear':emotion['fear'],'happy':emotion['happy'],'sad':emotion['sad'],'surprise':emotion['surprise'],'neutral':emotion['neutral'],'UE':user_eng,'CT':dt_string,'ST':startTime,'ET':endDateTime,'VT':videoTime,'FA':front_app,'PUE':previuos_UE}
                                }
                            }
                        T = threading.Thread(target=send_data(data))
                        T.start()
                        videoTime_old = int(videoTime)
                        // # filename = os.path.join(cwd,"brainalive","output","user_"+str(user_id),"video_"+str(ClassId), "images",str(videoTime) +".jpg")
                        // # os.makedirs(os.path.dirname(os.path.join(cwd, filename)), exist_ok=True) 
                        // # cv2.imwrite(filename, camframe,[int(cv2.IMWRITE_JPEG_QUALITY), 100])
                        if(previuos_UE_counter==2)
                            previuos_UE = user_eng
                    }
                        
                    
                } catch (error) {
                        console.log(`E9: ${error}`)
                }   
            } catch (error) {
                    console.log(`E10:${error}`)
            }


            let key = cv2.waitKey(1) & 0xFF
        
            if (key == ord("q"))
                break
            end = time.time()
            elapsed = end - start            


        } // while loop ends here


        cv2.destroyAllWindows()
        print("exit....")

    // code for finally
        window.localStorage.setItem('State',"Processing Complete");
        window.localStorage.setItem('Date',new Date());
    }catch(err){
        console.log(`E11:${err} Error in implementing the Algorithm`);
        window.localStorage.setItem('Error', err);
    }

}