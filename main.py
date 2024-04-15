import torch
import numpy as np

from logger import Logger
from collections import deque
from DQN.model import Model
from utils import get_random_image, preprocess_image, calculate_eps_decay


def test(model, logger):
    # Load the policy file
    model.network_local.load_state_dict(torch.load('checkpoint.pth'))
    model.network_local.eval()

    num_test_images = 0
    correct_predictions = 0

    for _ in range(num_test_images):
        # Get a random test image and its label
        image, true_label = get_random_image()

        # Preprocess the image
        processed_image = preprocess_image(image)

        # Send the image to the network for prediction
        predicted_label = model.predict(processed_image)

        if predicted_label == true_label:
            correct_predictions += 1

    accuracy = correct_predictions / num_test_images
    print(f'Test Accuracy: {accuracy:.2f}')
    logger.log_test_metrics(accuracy)

def train(model, logger, max_steps, eps_start, eps_end, eps_decay):
    eps = eps_start
    correct_predictions = 0
    total_loss = 0
    total_accuracy = 0
    total_magnitude = 0
    recent_accuracy = deque(maxlen=50)

    for step in range(1, max_steps + 1):
        # Get an image and its label at random
        image, true_label = get_random_image()

        # Preprocess the image
        processed_image = preprocess_image(image)

        model.update_beta((step - 1) / (max_steps - 1))

        # Send the image to the network
        predicted_label = model.predict(processed_image, eps)

        # Train the network and record metrics
        loss, accuracy, magnitude, lr = model.step(processed_image, true_label)

        total_loss += loss
        total_accuracy += accuracy
        total_magnitude += magnitude
        if predicted_label == true_label:
            correct_predictions += 1

        if eps > eps_end: eps -= eps_decay
        else: eps = eps_end

        if step % model.BATCH_SIZE == 0:
            batch_accuracy = correct_predictions / model.BATCH_SIZE
            recent_accuracy.append(batch_accuracy)
            avg_accuracy = np.mean(recent_accuracy)
            avg_loss = total_loss / model.BATCH_SIZE
            avg_magnitude = total_magnitude / model.BATCH_SIZE

            logger.log_metrics(step, eps, avg_accuracy, avg_loss, avg_magnitude, lr, model.prio_a, model.prio_b)
            
            print(f'\rStep: {step}\tEpsilon: {eps:.2f}\tBatch Accuracy: {batch_accuracy:.2f}\tAvg. Accuracy: {avg_accuracy:.2f}\tAvg. Loss: {avg_loss:.4f}\tAvg. Gradient Magnitude: {avg_magnitude:.4f}\tLearning Rate: {lr:.6f}', end='')
            if step % (model.BATCH_SIZE * 100) == 0:
                print(f'\rStep: {step}\tEpsilon: {eps:.2f}\tBatch Accuracy: {batch_accuracy:.2f}\tAvg. Accuracy: {avg_accuracy:.2f}\tAvg. Loss: {avg_loss:.4f}\tAvg. Gradient Magnitude: {avg_magnitude:.4f}\tLearning Rate: {lr:.6f}')
                torch.save(model.network_local.state_dict(), 'checkpoint.pth')

            correct_predictions = 0
            total_loss = 0
            total_accuracy = 0
            total_magnitude = 0

    torch.save(model.network_local.state_dict(), 'checkpoint.pth')
    return recent_accuracy

def main():
    load_policy = False
    max_steps = 1000000
    num_labels = 4
    eps_start = 1.0
    eps_end = 0.01
    eps_percentage = 0.98
    eps_decay = calculate_eps_decay(eps_start, eps_end, max_steps, eps_percentage)
    seed = 0
    priority_replay = [0.5, 0.5, 0.5]
    
    image_size = 0
    model = Model(image_size, num_labels, seed, priority_replay)
    logger = Logger(image_size, num_labels, eps_start, eps_end, eps_decay, max_steps)
    if not load_policy:
        _ = train(model, logger, max_steps, eps_start, eps_end, eps_decay)
    if load_policy:
        _ = test(model, logger)

    logger.close()

if __name__ == '__main__':
    main()
