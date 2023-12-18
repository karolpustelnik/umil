import cv2
import json
import os

train_path = '/workspace/chad_dataset/chad_meta/splits/train_split_2.txt'
videos_path = '/workspace/chad_dataset/chad/'
json_path = '/workspace/chad_dataset/chad_annotation/video_annotation/'
output_path = '/workspace/chad_dataset/chad_meta/splits/chad_train_umil.txt'

def get_class_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        return data['class']

def prepare_train(videos_path, json_path, train_path, output_path):
    open(output_path, 'w').close()
    f = open(train_path, 'r')
    for line in f:
        line = line.strip()
        video_path = videos_path + line
        print('Processing video:', video_path)
        # load video
        cap = cv2.VideoCapture(video_path + '.mp4')
        # get total number of frames
        total_frames = cap.get(cv2.CAP_PROP_FRAME_COUNT)
        if line.split('_')[2] == '0':
            category = '0'
        else:
            # check if json file exists
            path = json_path + line.split('.')[0] + '.json'
            if os.path.exists(path):
                category = get_class_from_json(path)
            else:
                # jump to next iteration
                continue
            
            
        with open(output_path, 'a') as file:
            
            file.write(line + ' ' + '0' +' ' + str(int(total_frames)) + ' ' + str(int(category)) + '\n')
        
prepare_train(videos_path, json_path, train_path, output_path)


test_path = '/workspace/chad_dataset/chad_meta/splits/test_split_2.txt'
videos_path = '/workspace/chad_dataset/chad/'
json_path = '/workspace/chad_dataset/chad_annotation/video_annotation/'
output_path = '/workspace/chad_dataset/chad_meta/splits/chad_test_umil.txt'

def get_class_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        return str(int(data['class']))
    
def get_anomaly_info_from_json(json_path):
    with open(json_path, 'r') as file:
        data = json.load(file)
        start_frame = data['start_frame']
        finish_frame = data['finish_frame']
        return (str(int(start_frame)), str(int(finish_frame)))
    
def prepare_test(videos_path, json_path, test_path, output_path):
    # clear file
    open(output_path, 'w').close()
    f = open(test_path, 'r')
    for line in f:
        line = line.strip()
        video_path = videos_path + line
        print('Processing video:', video_path)
        # load video
        cap = cv2.VideoCapture(video_path + '.mp4')
        # get total number of frames
        total_frames = str(int(cap.get(cv2.CAP_PROP_FRAME_COUNT)))
        
        if line.split('_')[2] == '0':
            start_frame =  '-1'
            finish_frame = '-1'
            category = '0'
        else:
            # check if json file exists
            path = json_path + line.split('.')[0] + '.json'
            if os.path.exists(path):
                start_frame, finish_frame = get_anomaly_info_from_json(path)
                category = get_class_from_json(path)
                
            else:
                # jump to next iteration
                continue
            
            
        with open(output_path, 'a') as file:
            
            file.write(line + ' ' + total_frames + ' ' + category + ' ' +   
            start_frame + ' ' + finish_frame + ' ' + '-1' + ' ' + '-1' '\n')
            
prepare_test(videos_path, json_path, test_path, output_path)