from ultralytics import YOLO
import cv2
import argparse

def main(args):
    # Load the exported engine
    model = YOLO(args.weights)
    
    # Open camera or video
    cap = cv2.VideoCapture(args.source)
    
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
            
        # Inference
        results = model.predict(frame, imgsz=args.imgsz, conf=args.conf)
        
        # Visualize
        annotated_frame = results[0].plot()
        
        cv2.imshow("Traffic Detection (Jetson Nano)", annotated_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
            
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="runs/train-finetune/weights/best.engine")
    parser.add_argument("--source", type=str, default="0", help="camera id or video path")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
