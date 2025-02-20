import torch
import torch.optim as optim
import numpy as np
from args import Args
from logger import Logger
from network import Network
from collections import deque
from utils import *

device = torch.device("cuda:0")

def test(images):
    args = Args()
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    network = Network(args)
    network.qnetwork_local.load_state_dict(torch.load(args.policy_file))
    network.qnetwork_local.to(device)
    network.qnetwork_local.eval()
    predictions = []
    
    with torch.no_grad():
        for image in images:
            # Process the image and move to correct device
            processed_image = preprocess_image(image, args.image_h, False)
            processed_image = processed_image.to(device)  # Move input to same device
            
            # Make prediction
            outputs = network.qnetwork_local(processed_image)
            predicted_label = torch.argmax(outputs).item()
            predictions.append(predicted_label)
            
            # Convert prediction to key press
            if predicted_label == 0:
                print("Left")
                APress()
            elif predicted_label == 1:
                print("Right")
                DPress()
            elif predicted_label == 2:
                print("Up")
                WPress()
            elif predicted_label == 3:
                print("Down")
                SPress()
            
            time.sleep(0.01)

    return predictions

def train_classifier(args, logger):
    network = Network(args)
    optimizer = optim.Adam(network.qnetwork_local.parameters(), lr=args.LR)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)
    criterion = torch.nn.CrossEntropyLoss()
    
    best_accuracy = 0
    scores_window = deque(maxlen=100)
    
    for step in range(1, args.max_steps + 1):
        network.qnetwork_local.train()
        
        # Get batch of images and labels
        images, labels = get_batch_of_images(args.image_dir, args.BATCH_SIZE)
        
        # Process images and move to correct device
        processed_images = torch.stack([preprocess_image(img, args.image_h, False) for img in images]).to(device)
        labels = torch.tensor(labels, dtype=torch.long).to(device)
        
        if step == 1:  # First batch only
            print("Sample batch stats:")
            print(f"Processed images shape: {processed_images.shape}")
            print(f"Processed images min/max: {processed_images.min()}, {processed_images.max()}")
            print(f"Labels: {labels}")
        # Forward pass
        optimizer.zero_grad()
        outputs = network.qnetwork_local(processed_images)
        loss = criterion(outputs, labels)
        
        # Backward pass
        loss.backward()
        optimizer.step()
        
        # Calculate accuracy
        predictions = torch.argmax(outputs, dim=1)
        accuracy = (predictions == labels).float().mean().item()
        scores_window.append(accuracy)
        
        # Logging
        if logger is not None and step % 1000 == 0:
            avg_accuracy = np.mean(scores_window)
            current_lr = optimizer.param_groups[0]['lr']
            logger.log_metrics(step, loss.item(), accuracy, avg_accuracy, current_lr)
            print(f'\rStep: {step}\tLoss: {loss.item():.4f}\tAccuracy: {accuracy:.4f}\tAverage Accuracy: {avg_accuracy:.4f}\tLR: {current_lr:.6f}')
            
            # Save best model
            if avg_accuracy > best_accuracy:
                best_accuracy = avg_accuracy
                torch.save(network.qnetwork_local.state_dict(), 'best_model.pth')
        
        # Update learning rate
        if step % 1000 == 0:
            scheduler.step(loss)
    
    # Save final model
    torch.save(network.qnetwork_local.state_dict(), 'final_model.pth')
    
    return scores_window

def get_batch_of_images(image_dir, batch_size):
    images = []
    labels = []
    for _ in range(batch_size):
        image_path, label = get_random_image(image_dir)
        images.append(cv2.imread(image_path))
        labels.append(label)
    return images, labels
