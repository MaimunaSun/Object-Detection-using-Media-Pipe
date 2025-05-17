import cv2
import mediapipe as mp
import os
import pandas as pd
from datetime import datetime, timezone

# Function to extract the frames from the walk video and get te frames per second (FPS)
def extract_frames(video, frames_folder):
    #open walk video file
    video_capture = cv2.VideoCapture(video)
    
    #get the fps
    FPS = video_capture.get(cv2.CAP_PROP_FPS)
    print(f"The walk video video FPS is: {FPS}")

    #creating directory for the frames
    os.makedirs(frames_folder, exist_ok=True)

    frame_number = 0

    #extrcting the frames and saving them in the directory
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        if not ret:
            break

        frame_name = f'{frames_folder}/frame_{frame_number:04d}.jpg'
        cv2.imwrite(frame_name, frame)
        frame_number += 1
    
    #release capture object
    video_capture.release()
    #return the fps and the total frame numer extrcated
    print(f"Extracted {frame_number} frames and saved them in '{frames_folder}'.")
    return FPS, frame_number

# Function to detect the initial heel contact frame in the walk video
def Video_Heel_Contact(frames_folder, FPS):
    #initilize mediapipe pose class
    mp_pose = mp.solutions.pose
    #setting up pose fnction for frame images
    pose = mp_pose.Pose(static_image_mode=True)

    #sorting the frames in the directory
    Sorted_frames = sorted([f for f in os.listdir(frames_folder) if f.endswith('.jpg')])

    #initilizing frame of initial heel contact
    heel_contact_frame = None

    for frame_number, frame_name in enumerate(Sorted_frames):
        #get the path of the frame
        frame_path = os.path.join(frames_folder, frame_name)
        #load the image frame
        image = cv2.imread(frame_path)
        #convert image frame from BGR to RGB for pose detection
        image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        #pose detection using media pipe 
        results = pose.process(image_rgb)

        #if pose landmarks are detected on the image
        if results.pose_landmarks:
            #get the pose landmarks
            landmarks = results.pose_landmarks.landmark
            #get the ankle landamrk
            left_ankle = landmarks[mp_pose.PoseLandmark.LEFT_ANKLE]
            #get the foot index landamrk
            left_foot_index = landmarks[mp_pose.PoseLandmark.LEFT_FOOT_INDEX]
            #if the foot is at a higher position than ankle than heel contact detected
            if left_ankle.y > left_foot_index.y:
                #Set heel contact frame to the fram number
                heel_contact_frame = frame_number
                print(f"Initial Heel contact detected in frame: {frame_number} ({frame_name})")
                break  

    #if the heel contact frame is detected return heel contact frame and heel contact time
    if heel_contact_frame is not None:
        #Get the initial heel contact time from the video
        heel_contact_time = heel_contact_frame / FPS
        print(f"Initial Heel contact in video at: {heel_contact_time:.2f} seconds")
        return heel_contact_time, heel_contact_frame
    else:
        print("No initial heel contact detected.")
        return None, None
    
#Function to detect the initial heel contact from the accelrometer data
def Accelerometer_Heel_Contact(accelerometer_data):
    #load csv file in to a dataframe
    df = pd.read_csv(accelerometer_data)
    #get the utc_time from the acceleromter data
    df['utc_time'] = df['utc_time'].astype(float)

    #get the maximum acceleration
    maximum_acceleration = df['accel_x[counts]'].idxmax()
    #get time of the maximum acceleration
    heel_contact_time = df['utc_time'][maximum_acceleration]

    #convert the utc time to readbale format
    readable_time = datetime.fromtimestamp(heel_contact_time, tz=timezone.utc).strftime('%H:%M:%S')
    print(f"Initial heel contact detected at {readable_time} (UTC Time: {heel_contact_time}) from accelerometer data")

    return heel_contact_time

#Function to sync accelrometer data with walk video
def Syncing_Tool(video, accelerometer_data):
    #Folder with the walk video frames
    frames_folder = 'frames'

    #Extract frames from the walk video
    FPS, total_frames = extract_frames(video, frames_folder)

    #Calculate walk video duration
    video_duration = total_frames / FPS

    #Detect heel contact time and heel contact frame in the walk video
    video_heel_contact_time, heel_contact_frame = Video_Heel_Contact(frames_folder, FPS)

    #Time from heel contact to start of the video and end of the video
    heel_contact_to_start_video = video_heel_contact_time 
    heel_contact_to_end_video = video_duration - video_heel_contact_time
    
    #Detect heel contact time in the accelerometer data
    accelerometer_heel_contact_time = Accelerometer_Heel_Contact(accelerometer_data)

    #Load accelerometer data and get the start time and end time
    df = pd.read_csv(accelerometer_data)
    accelerometer_start_time = df['utc_time'].iloc[0]
    accelerometer_end_time = df['utc_time'].iloc[-1]

    #Time from heel contact to start and end of the accelerometer data
    heel_contact_to_start_accelerometer = accelerometer_heel_contact_time - accelerometer_start_time
    heel_contact_to_end_accelerometer = accelerometer_end_time - accelerometer_heel_contact_time

    #Find the smallest time duration between heel contact time and end of acceleromter data and end of video
    trim_duration_forward = min(heel_contact_to_end_video, heel_contact_to_end_accelerometer)
    
    #Find the smallest time duration between heel contact time and start of acceleromter data and start of video
    trim_duration_backward = min(heel_contact_to_start_video, heel_contact_to_start_accelerometer)

    #Calculate the new start and end points for trimming in the video
    new_video_start_frame = heel_contact_frame - trim_duration_backward * FPS
    new_video_end_frame = heel_contact_frame + trim_duration_forward * FPS

    #Calculate the new start and end points for trimming in the acceleomter data
    new_accelerometer_start_time = accelerometer_heel_contact_time - trim_duration_backward
    new_accelerometer_end_time = accelerometer_heel_contact_time + trim_duration_forward

    #Trim video and accelerometer data
    trim_video(video, new_video_start_frame, new_video_end_frame, FPS)
    trim_accelerometer_data(accelerometer_data, new_accelerometer_start_time, new_accelerometer_end_time)

#Function to trim the video
def trim_video(video, start_frame, end_frame, FPS):
    #OPEN WALK VIDEO
    video_capture = cv2.VideoCapture(video)
    #GET THE FRAME WIDTH
    frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
    #GET THE FRAME HEIGHT
    frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

    #writing frames to trimmed video
    video_writer = cv2.VideoWriter('trimmed_Walk.mp4', cv2.VideoWriter_fourcc(*'mp4v'), FPS, (frame_width, frame_height))
    
    frame_number = 0
    while video_capture.isOpened():
        ret, frame = video_capture.read()
        
        if not ret:
            break 

        if start_frame <= frame_number < end_frame:
            video_writer.write(frame)
        
        frame_number += 1


    #release capture and writer objects
    video_capture.release()
    video_writer.release()
    print(f"Trimmed video saved as 'trimmed_Walk.mp4'")

#Function to trim accelerometer data
def trim_accelerometer_data(accelerometer_data, start_time, end_time):
    #load csv file in to a dataframe
    df = pd.read_csv(accelerometer_data)
    #get the utc_time from the acceleromter data
    df['utc_time'] = df['utc_time'].astype(float)

    # Trim the data to the desired time range
    trimmed_df = df[(df['utc_time'] >= start_time) & (df['utc_time'] <= end_time)]
    trimmed_df.to_csv('trimmed_ACCEL.csv', index=False)
    print(f"Trimmed accelerometer data saved as 'trimmed_ACCEL.csv'")

#Run the Syncing tool
video = 'Walk.MP4' 
acceleromter_data = 'ACCEL.csv' 
Syncing_Tool(video, acceleromter_data)
