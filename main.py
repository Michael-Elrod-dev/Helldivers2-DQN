import torch
import numpy as np

from args import Args
from logger import Logger
from collections import deque
from DQN.network import Network
from utils import get_random_image, preprocess_image


def test(args, network, logger):
    # Load the policy file
    network.network_local.load_state_dict(torch.load('checkpoint.pth'))
    network.network_local.eval()

    num_test_images = 0
    correct_predictions = 0

    for _ in range(num_test_images):
        # Get a random test image and its label
        image, true_label = get_random_image()

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Send the image to the network for prediction
        predicted_label = network.predict(processed_image)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_images
    print(f'Test Accuracy: {accuracy:.2f}')
    if logger: logger.log_test_metrics(accuracy)

def train(args, network, logger):
    eps = args.eps_start
    correct_predictions = 0
    total_loss = 0
    total_accuracy = 0
    total_magnitude = 0
    recent_accuracy = deque(maxlen=50)

    torch.set_printoptions(threshold=10_000)
    for step in range(1, args.max_steps + 1):
        # Get an image and its label at random
        image, true_label = get_random_image(args.image_dir)
        # print(image, true_label)
        
        # Preprocess the image
        processed_image = preprocess_image(image, args.image_w, args.image_h)
        print(processed_image.shape)

        network.update_beta((step - 1) / (args.max_steps - 1))

        # Send the image to the network
        predicted_label = network.predict(processed_image, eps)

        # Train the network and record metrics
        loss, accuracy, magnitude, lr = network.step(processed_image, true_label)

        total_loss += loss
        total_accuracy += accuracy
        total_magnitude += magnitude
        if predicted_label == true_label:
            correct_predictions += 1

        if eps > args.eps_end: eps -= args.eps_decay
        else: eps = args.eps_end

        if step % args.BATCH_SIZE == 0:
            batch_accuracy = correct_predictions / args.BATCH_SIZE
            recent_accuracy.append(batch_accuracy)
            avg_accuracy = np.mean(recent_accuracy)
            avg_loss = total_loss / args.BATCH_SIZE
            avg_magnitude = total_magnitude / args.BATCH_SIZE

            if logger: logger.log_metrics(step, eps, avg_accuracy, avg_loss, avg_magnitude, lr, args.prio_a, args.prio_b)
            
            print(f'\rStep: {step}\tEpsilon: {eps:.2f}\tBatch Accuracy: {batch_accuracy:.2f}\tAvg. Accuracy: {avg_accuracy:.2f}\tAvg. Loss: {avg_loss:.4f}\tAvg. Gradient Magnitude: {avg_magnitude:.4f}\tLearning Rate: {lr:.6f}', end='')
            if step % (args.BATCH_SIZE * 100) == 0:
                print(f'\rStep: {step}\tEpsilon: {eps:.2f}\tBatch Accuracy: {batch_accuracy:.2f}\tAvg. Accuracy: {avg_accuracy:.2f}\tAvg. Loss: {avg_loss:.4f}\tAvg. Gradient Magnitude: {avg_magnitude:.4f}\tLearning Rate: {lr:.6f}')
                torch.save(args.network_local.state_dict(), 'checkpoint.pth')

            correct_predictions = 0
            total_loss = 0
            total_accuracy = 0
            total_magnitude = 0

    torch.save(network.network_local.state_dict(), 'checkpoint.pth')
    return recent_accuracy

def main():
    args = Args()
    
    network = Network(args)
    if args.wandb: 
        logger = Logger(args)
    else: 
        logger = None

    if not args.load_policy:
        _ = train(args, network, logger)
    if args.load_policy:
        _ = test(args, network, logger)

    if args.wandb: logger.close()

if __name__ == '__main__':
    main()
