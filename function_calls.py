from function_definitions import YoLov5TRT, YoloTRT, CustomYoloClass
import sys
import cv2
import ctypes 
import time

def integration(cls_flag=0):
    PLUGIN_LIBRARY = "./libmyplugins.so"
    engine_file_path = "./best_seg.engine"
    if (cls_flag == 1):
        cls_engine_file_path = "./best_cls.engine"
    model = YoloTRT(library="./libmyplugins.so", engine="./TF.engine", conf=0.5, yolo_ver="v5")
    if len(sys.argv) > 1:
        engine_file_path = sys.argv[1]
    if len(sys.argv) > 2:
        PLUGIN_LIBRARY = sys.argv[2]

    ctypes.CDLL(PLUGIN_LIBRARY)

    categories = ["Footpath","Road"]
    

    # Create an instance of the YoLov5TRT class
    print(1)
    yolov5_wrapper = YoLov5TRT(engine_file_path)
    print(2)
    if (cls_flag == 1):
        cls_yolov5_wrapper = CustomYoloClass(cls_engine_file_path)
    print(3)

    # Open a video capture object
    video_path = "./demo.mp4"  # Replace with your video file path
    print(4)
    cap = cv2.VideoCapture(0,cv2.CAP_V4L2)
    fps_arr = []
    cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
    cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 640)
    print(5)
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Resize the frame if needed
            frame = cv2.resize(frame, (640, 640))
            #frame = cv2.cvtColor(frame,cv2.COLOR_BGR2GRAY)

            # Perform inference on the current frame - segmentation
            starting = time.time()
            result_image, use_time, overlap_road, overlap_footpath,resultant_prediction = yolov5_wrapper.infer([frame])
            ending = time.time()
            print(f'Segmentation time = {ending-starting} Sec')
            
            
            starting1 = time.time()
            detections, t = model.Inference(frame) 
            ending1 = time.time()
            print(f'Detection time = {ending1-starting1} Sec')
            
            if (cls_flag == 1):
                starting2 = time.time()
                height = frame.shape[0]
                cropped = frame[int(height / 2):]
                cropped = cv2.resize(cropped, (640, 640))
                result_image, prediction, use_time = cls_yolov5_wrapper.infer([cropped])
                print(f"{prediction}")
                ending2 = time.time()
                print(f'Classification Time = {ending2 - starting2}')
                
            print(f'Final time = {ending2-starting} Sec')
            font = cv2.FONT_HERSHEY_SIMPLEX
            org = (50, 50)
            fontScale = 1
            color = (255, 0, 0)
            thickness = 2
            result_image[0] = cv2.putText(result_image[0], resultant_prediction, org, font, fontScale, color, thickness, cv2.LINE_AA)
            # Display or save the processed frame
            #cv2.imshow("Processed Frame", result_image[0])
            cv2.imshow("Processed Frame",result_image[0])
            fps_arr.append(ending1-starting)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(e)

    finally:
        #mean = sum(fps_arr)/len(fps_arr)
        #print(f'Mean FPS: {1/mean}')
        cap.release()
        cv2.destroyAllWindows()
        yolov5_wrapper.destroy()
integration()
