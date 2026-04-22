from ultralytics import YOLO
import argparse
import os

def main(args):
    # Load model
    # Note: If testing before training, use 'yolov8n.pt'
    model = YOLO(args.weights)
    
    # Run inference
    # YOLO handles camera, video, image, or folder automatically via 'source'
    print(f"Running inference on: {args.source}")
    results = model.predict(
        source=args.source, 
        imgsz=args.imgsz, 
        conf=args.conf, 
        save=True,           # Save results to runs/detect
        show=False           # Set to True if you have a monitor
    )
    
    print(f"\nInference completed. Results saved to 'runs/detect'")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--weights", type=str, default="yolov8n.pt", help="model.pt or model.engine")
    parser.add_argument("--source", type=str, default="0", help="camera id, video path, or image folder")
    parser.add_argument("--imgsz", type=int, default=640)
    parser.add_argument("--conf", type=float, default=0.25)
    args = parser.parse_args()
    main(args)
