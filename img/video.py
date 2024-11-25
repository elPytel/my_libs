import cv2 as cv

def get_frames(video_path, frame_numbers : list):
    video = cv.VideoCapture(video_path)

    frames = []
    index = 0
    try:
        while(video.isOpened()):
            ret, frame = video.read()
            if not ret:
                break
            
            if index in frame_numbers:
                frames.append(frame)
                frame_numbers.remove(index)
                if len(frame_numbers) == 0:
                    break
            index += 1
    except KeyboardInterrupt:
        pass
    finally:
        video.release()
    return frames