# This file will preprocess raw .mp4 videos into individual frames
import cv2
import os


def get_videos(dir):
    # assign directory
    directory = dir
    
    # iterate over files in
    # that directory
    path_list = []
    for filename in os.listdir(directory):
        f = os.path.join(directory, filename)
        # checking if it is a file
        
        if os.path.isfile(f):
            print(f)
            path_list.append(f)
    return path_list

def resize_image(img):
    # Resize image to 256x256
    width = 256
    height = 256
    dim = (width, height)
    resized = cv2.resize(img, dim, interpolation = cv2.INTER_AREA)
    return resized

def main():
    videos = get_videos("./raw_timelapse")
    print(type(videos))
    print(videos)
    print("Number of videos: {}".format(len(videos)))
    for vidnum, videoname in enumerate(videos):
        vidcap = cv2.VideoCapture(videoname)
        success,image = vidcap.read()
        count = 0
        while success:
            cv2.imwrite("./gan_data/clouds/v{}frame{}.jpg".format(vidnum, count), resize_image(image))     # save frame as JPEG file      
            success,image = vidcap.read()
            count += 1
        print("Video {}: Complete".format(vidnum))
    print("All videos converted to images")

if __name__ == "__main__":
    main()