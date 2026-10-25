import cv2

def process_video(video_path):
    cap = cv2.VideoCapture(video_path)

    if not cap.isOpened():
        print("Error: Unable to open video.")
        return
    while True:
        ret, frame=cap.read()
        if not ret:
            break
        cv2.imshow("Safe Turn Prdictor",frame)
        if cv2.waitKey(25) & 0xFF == ord('q'):
            break
    cap.release()
    cv2.destroyAllWindows()
if __name__=="__main__":
    process_video("datasets/videos/test_video1.mp4")
        