import numpy as np
import timeit
import os
from skimage import io
from skimage.draw import polygon_perimeter
import dlib
import cv2
from pathlib import Path
from tqdm import tqdm
import shutil
import argparse
import json

def extract_frames(video_path, sample_ratio=30):

    # output
    output_frames = []
    output_frame_id = []

    # define a video capture object
    vid = cv2.VideoCapture(video_path)

    totalFrames = int(vid.get(cv2.CAP_PROP_FRAME_COUNT))
    x = [i for i in range (1, totalFrames) if divmod(i, int(sample_ratio))[1]==0]
    
    for myFrameNumber in tqdm(x):
        # set your counter to specific frame that u want to get
        vid.set(cv2.CAP_PROP_POS_FRAMES, myFrameNumber)
        
        # read the frame
        ret, frame = vid.read()
        
        # if it's readable
        if ret == True:
            output_frames.append(frame) 
            output_frame_id.append(myFrameNumber) 

    return (output_frames, output_frame_id)

def face_detection(face_detector, image_list, frame_num_list, upsample_ratio=2, batch_size=64,
                   video_name="Unnamed_Video", algorithm_name = "Unnamed_Algorithm", save_folder_path = None):

    # detection per image meta-data
    output_dict = {}
    output_dict["VideoShape"] = image_list[0].shape
    output_dict["FaceExtractor"] = algorithm_name
    output_dict["VideoName"] = video_name


    dets = face_detector(image_list, upsample_ratio, batch_size = batch_size)

    for det, image, frame_num in zip(dets, image_list, frame_num_list):
        
        # means that there is no face 
        if len(det) <= 0:
            continue
        
        else:

            image_original = image.copy()

            # saving params.
            bbox_coords = []
            save_name = video_name+"_"+str(frame_num)+".jpg"
            
            # draw bb to original frame
            for d in det:
                rr,cc = polygon_perimeter([d.rect.top(), d.rect.top(), d.rect.bottom(), d.rect.bottom()],
                                    [d.rect.right(), d.rect.left(), d.rect.left(), d.rect.right()])
                image[rr, cc] = (255, 0, 0)

                # cropping coords
                x = np.maximum(d.rect.left(), 0)
                y = np.maximum(d.rect.top(), 0)
                w = np.minimum(d.rect.right() - x, image.shape[0])
                h = np.minimum(d.rect.bottom() - y, image.shape[1])

                
                bbox_coords.append((int(x),int(y),int(w),int(h)))
                
                # crop only face image
                #cropped_image = image_original[y:y+h, x:x+w]

            # to see what about the bounding boxes for faces
            #io.imsave(os.path.join(os.path.join(save_folder_path, video_name), video_name+"_"+str(frame_num)+".jpg"), image[:,:,::-1])

            # save without any annotations
            io.imsave(os.path.join(os.path.join(save_folder_path, video_name), save_name), image_original[:,:,::-1])

            # output dict append new frame info
            output_dict[save_name] = bbox_coords

    

    with open(os.path.join(os.path.join(save_folder_path, video_name), video_name+"_label.json"), 'w') as f:
        json.dump(output_dict, f)


def main(Algo_Name, Algo_Weights, Sample_Ratio, Upsampeled_Ratio, Input_Path, Save_Path):

    print(Algo_Name, Algo_Weights, Sample_Ratio, Upsampeled_Ratio, Input_Path, Save_Path)

    # Take the video
    # Sample 1 frame per second
    # Then according to the face existance in the frames, decide to save or not

    face_detector_name = Algo_Name

    # dlib constr.
    dlib_face_detector_path = Algo_Weights
    cnn_face_detector = dlib.cnn_face_detection_model_v1(dlib_face_detector_path)    

    # video lists
    video_list = os.listdir(Input_Path)

    # for each video;
    #   a) Extract Frames
    #   b) Find Faces etc.
    for video in video_list:
        vid_name = Path(os.path.join(Input_Path,video)).stem
        
        # create output directory
        if not os.path.exists(os.path.join(Save_Path, vid_name)):
            os.makedirs(os.path.join(Save_Path, vid_name))
        else:
            shutil.rmtree(os.path.join(Save_Path, vid_name), ignore_errors=True)
            os.makedirs(os.path.join(Save_Path, vid_name))


        print(vid_name +" is being processed...")

        # video frame extraction
        video_frames, video_frame_ids = extract_frames(os.path.join(Input_Path, video), sample_ratio=Sample_Ratio)        

        # frame face detection
        start = timeit.default_timer()
        face_detection(face_detector = cnn_face_detector, image_list= video_frames, frame_num_list=video_frame_ids, upsample_ratio=Upsampeled_Ratio,
                       batch_size=1, video_name=vid_name, algorithm_name = face_detector_name, save_folder_path = Save_Path)
        stop = timeit.default_timer()
        print('Time for face detection: ', stop - start)  


if __name__ == "__main__":

    # Create the parser and add arguments
    parser = argparse.ArgumentParser("Hey! The most simplified usage is; put the input videos in the videos folder which is located at one above directory then type: 'python face_detector.py'")
    parser.add_argument('-an', dest='Algo_Name', help="Face Detector Algorithm Name", default="Dlib - CNN")
    parser.add_argument('-aw', dest="Algo_Weights", help="Dlib CNN Model's Weight", default="../detector_weight/mmod_human_face_detector.dat")
    parser.add_argument('-sr', dest='Sample_Ratio', help="Video Sample Ratio by Frame, e.g. 1000 refers to select every 1000th frame in the video", default=1000)
    parser.add_argument('-ur', dest='Upsampeled_Ratio', help="Dlib Face Detector, Upsample Ratio", default=1)
    parser.add_argument('-ip', dest='Input_Path', help="Input Data Folder Path", default="../videos")
    parser.add_argument('-up', dest='Save_Path', help="Output Data Folder Path", default= "../face_dataset/")

    # Parse the arguments
    args = vars(parser.parse_args())

    # main funtion
    main(args['Algo_Name'],args['Algo_Weights'],args['Sample_Ratio'],args['Upsampeled_Ratio'],args['Input_Path'],args['Save_Path'])