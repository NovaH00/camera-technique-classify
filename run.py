import os
import cv2
import torch
import numpy as np
import argparse
from PIL import Image
from torchvision import transforms
from model import VideoClassify
import time



def load_model(model_path, num_classes=5, frames=75):
    """
    Load a trained model from checkpoint.
    
    Args:
        model_path (str): Path to the model file
        num_classes (int): Number of classes the model was trained on
        frames (int): Number of frames the model expects
        
    Returns:
        tuple: (model, device, class_map)
    """
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Initialize the model
    model = VideoClassify(num_classes=num_classes, frames=frames)
    
    # Load model
    if os.path.isfile(model_path):
        print(f"Loading model from {model_path}")
        checkpoint = torch.load(model_path, map_location=device)  # Use 'checkpoint' instead of overwriting 'model'
        
        # Load state dict from checkpoint
        if 'model_state_dict' in checkpoint:
            model.load_state_dict(checkpoint['model_state_dict'])
            print(f"Model loaded successfully. Best validation accuracy: {checkpoint.get('best_val_acc', 'N/A'):.4f}")
        else:
            model.load_state_dict(checkpoint)
            print("Model loaded successfully.")
            
        # Extract class_map from checkpoint if available
        class_map = None
        if 'class_map' in checkpoint:
            original_class_map = checkpoint['class_map']
            print("Raw class map from checkpoint:", original_class_map)
            
            # Convert to idx -> name format if necessary
            # Check if keys are strings (class names) and values are indices
            if all(isinstance(k, str) for k in original_class_map.keys()) and all(isinstance(v, int) for v in original_class_map.values()):
                # Invert the map to have indices as keys and class names as values
                class_map = {v: k for k, v in original_class_map.items()}
                print("Converted class map (idx -> name):", class_map)
            else:
                class_map = original_class_map
                
        else:
            print("No class map found in checkpoint, using default class map.")
            class_map = {
                0: "dolly",
                1: "pan",
                2: "tilt", 
                3: "tracking", 
                4: "zoom"
            }
    else:
        raise FileNotFoundError(f"No model found at {model_path}")
    
    model.to(device)
    model.eval()
    return model, device, class_map

def preprocess_frame(frame, transform=None):
    """
    Preprocess a single video frame for inference.
    
    Args:
        frame: CV2 frame (BGR format)
        transform: Optional transform to apply
        
    Returns:
        torch.Tensor: Preprocessed frame tensor
    """
    if transform is None:
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    # Convert BGR to RGB
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    frame_pil = Image.fromarray(frame_rgb)
    
    # Apply transformations
    return transform(frame_pil)

def sample_frames(video_path, num_frames=75):
    """
    Sample frames at even intervals from a video.
    
    Args:
        video_path (str): Path to the video file
        num_frames (int): Number of frames to sample
            
    Returns:
        list: List of sampled frames as tensors
    """
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        raise ValueError(f"Could not open video: {video_path}")
    
    # Get video properties
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    if total_frames <= 0:
        raise ValueError(f"Video has no frames: {video_path}")
    
    # Calculate frame indices to sample
    if total_frames <= num_frames:
        # If video has fewer frames than needed, duplicate frames
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    else:
        # Sample frames at even intervals
        indices = np.linspace(0, total_frames - 1, num_frames, dtype=int)
    
    # Create transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    frames = []
    for idx in indices:
        # Set frame position
        cap.set(cv2.CAP_PROP_POS_FRAMES, idx)
        ret, frame = cap.read()
        
        if not ret:
            # If frame read failed, create a black frame
            height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
            width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
            frame = np.zeros((height, width, 3), dtype=np.uint8)
        
        # Preprocess frame
        processed_frame = preprocess_frame(frame, transform)
        frames.append(processed_frame)
    
    cap.release()
    return frames

def run_inference(model, video_path, device, class_map=None, frames=75):
    """
    Run inference on a video file.
    
    Args:
        model: PyTorch model
        video_path (str): Path to the video file
        device: PyTorch device
        class_map (dict): Mapping from class index to class name
        frames (int): Number of frames to sample
        
    Returns:
        tuple: (predicted_class_idx, prediction_scores)
    """
    # Default class map if none provided
    if class_map is None:
        class_map = {
            0: "dolly",
            1: "pan",
            2: "tilt", 
            3: "tracking", 
            4: "zoom"
        }
    
    # Sample frames from video
    print(f"Sampling {frames} frames from video...")
    sampled_frames = sample_frames(video_path, num_frames=frames)
    
    # Stack frames and add batch dimension
    inputs = torch.stack(sampled_frames).unsqueeze(0)  # [1, T, C, H, W]
    inputs = inputs.to(device)
    
    # Get predictions
    print("Running inference...")
    with torch.no_grad():
        outputs = model(inputs)
        probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
    # Get predicted class
    predicted_class = torch.argmax(probabilities).item()
    
    # Print results
    print("\nPrediction Results:")
    print("-" * 50)
    for class_idx, class_name in class_map.items():
        # Ensure class_idx is an integer
        class_idx_int = int(class_idx)
        # Check if index is within bounds
        if 0 <= class_idx_int < len(probabilities):
            probability = probabilities[class_idx_int].item() * 100
            print(f"{class_name}: {probability:.2f}%")
        else:
            print(f"{class_name}: Index {class_idx_int} out of bounds")
    print("-" * 50)
    
    # Get predicted class name (safely)
    if predicted_class in class_map:
        print(f"Predicted camera technique: {class_map[predicted_class]}")
    else:
        print(f"Predicted class index {predicted_class} not found in class map")
    
    # Return both predicted class and all probabilities
    return predicted_class, probabilities.cpu().numpy()

def visualize_result(video_path, predicted_class, probabilities=None, class_map=None, output_path=None, true_class=None, resize_factor=0.5):
    """
    Visualize the prediction result on the video.
    
    Args:
        video_path (str): Path to the input video
        predicted_class (int): Predicted class index
        probabilities (numpy.ndarray): Array of probabilities for each class
        class_map (dict): Mapping from class index to class name
        output_path (str): Path to save the output video
        true_class (str): Optional true class name for evaluation
        resize_factor (float): Factor to resize the frame (0.5 = 50%)
    """
    if class_map is None:
        class_map = {
            0: "dolly",
            1: "pan",
            2: "tilt", 
            3: "tracking", 
            4: "zoom"
        }
    
    # Convert class_map keys to int if they're strings
    class_map = {int(k) if isinstance(k, str) else k: v for k, v in class_map.items()}
    
    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"Error opening video file {video_path}")
        return
    
    # Get video properties
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = 15
    
    # Calculate new dimensions (50% of original size)
    new_width = int(width * resize_factor)
    new_height = int(height * resize_factor)
    
    # Create video writer if output path is specified
    writer = None
    if output_path:
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(output_path, fourcc, fps, (new_width, new_height))
    
    # Get predicted technique (safely)
    predicted_technique = class_map.get(predicted_class, f"Unknown (Class {predicted_class})")
    
    # Sort probabilities for display if provided
    sorted_probs = []
    if probabilities is not None:
        # Create (class_idx, probability) pairs and sort by probability (descending)
        sorted_probs = [(i, probabilities[i]) for i in range(len(probabilities)) if i in class_map]
        sorted_probs.sort(key=lambda x: x[1], reverse=True)
    
    while cap.isOpened():
        start_time = time.time()
        
        ret, frame = cap.read()
        if not ret:
            break
        
        
        # Resize the frame to 50% of original size
        frame = cv2.resize(frame, (new_width, new_height))
            
        # Add title
        cv2.putText(
            frame, 
            "Camera Technique Classification",
            (10, 30), 
            cv2.FONT_HERSHEY_SIMPLEX, 
            0.9, 
            (255, 255, 255), 
            2
        )
        
        # Display true class if provided
        if true_class:
            cv2.putText(
                frame,
                f"True Class: {true_class}",
                (10, new_height - 20),
                cv2.FONT_HERSHEY_SIMPLEX,
                0.7,
                (255, 255, 0),  # Yellow color for true class
                2
            )
        
        # Display probabilities in order
        y_offset = 70
        for i, (class_idx, prob) in enumerate(sorted_probs):
            class_name = class_map[class_idx]
            probability_text = f"{class_name}: {prob * 100:.2f}%"
            
            # Green for highest probability, red for others
            color = (0, 255, 0) if i == 0 else (0, 0, 255)
            
            cv2.putText(
                frame, 
                probability_text,
                (10, y_offset + i * 30), 
                cv2.FONT_HERSHEY_SIMPLEX, 
                0.7, 
                color, 
                2
            )
        
        # Show the frame
        cv2.imshow('Camera Technique Classification', frame)
        
        # Write frame if writer exists
        if writer:
            writer.write(frame)
            
        # Press Q to exit
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break
        
        time_elapse = time.time() - start_time
        
        delta_time = max(0, 1/fps - time_elapse)
        
        time.sleep(delta_time)
    
    # Release resources
    cap.release()
    if writer:
        writer.release()
    cv2.destroyAllWindows()
    
    if output_path:
        print(f"Output video saved to {output_path}")

def main():
    parser = argparse.ArgumentParser(description='Camera Technique Classification')
    parser.add_argument('--video', type=str, required=True, help='Path to the input video')
    parser.add_argument('--model', type=str, default='./checkpoints/best_model.pth', 
                        help='Path to the model checkpoint')
    parser.add_argument('--num_classes', type=int, default=5, help='Number of classes')
    parser.add_argument('--frames', type=int, default=75, help='Number of frames to sample')
    parser.add_argument('--output', type=str, help='Path to save the output video')
    parser.add_argument('--visualize', action='store_true', help='Visualize the results')
    parser.add_argument('--true_class', type=str, help='True class name for evaluation')
    parser.add_argument('--resize', type=float, default=0.5, help='Factor to resize the frame (0.5 = 50%)')
    
    args = parser.parse_args()
    
    # Load model and get class map from checkpoint
    model, device, class_map = load_model(args.model, num_classes=args.num_classes, frames=args.frames)
    
    # Run inference
    predicted_class, probabilities = run_inference(model, args.video, device, class_map=class_map, frames=args.frames)
    
    # Visualize result if requested
    if args.visualize or args.output:
        visualize_result(args.video, predicted_class, probabilities=probabilities, 
                         class_map=class_map, output_path=args.output,
                         true_class=args.true_class, resize_factor=args.resize)

if __name__ == "__main__":
    main()