import torch
import numpy as np
from args import Args
from logger import Logger
from dqn_agent import Agent
from collections import deque
from utils import get_random_image, preprocess_image, env_step, rename_file


def test_dqn(args, logger):
    network = Agent(args)
    network.qnetwork_local.load_state_dict(torch.load('learned_policy.pth'))

    scores_window = deque(maxlen=100)
    scores = []
    eps = 0
    
    for step in range(1, args.test_steps + 1):
        reward = 0
        # Get an image path and its label at random
        image_path, true_label = get_random_image(args.image_dir)
        
        # Process the image
        processed_image = preprocess_image(image_path, args.image_h)
        processed_image = torch.flatten(processed_image)
        
        predicted_label = network.act(processed_image, eps)
        next_image, reward, done = env_step(predicted_label, true_label, processed_image)
        
        scores_window.append(reward)
        scores.append(reward)
        
        print('Step: {}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(step, reward, np.mean(scores_window)))

    # Calculate correct and incorrect predictions
    correct = scores.count(5)
    incorrect = scores.count(0)

    # Print the results
    print(f'Number of correct labels: {correct}')
    print(f'Number of incorrect labels: {incorrect}')

    return scores

def dqn(args, logger):
    network = Agent(args)
    scores = []
    eps = args.eps_start
    scores_window = deque(maxlen=100)

    for step in range(1, args.max_steps + 1):
        reward = 0
        # Get an image path and its label at random
        image_path, true_label = get_random_image(args.image_dir)
        
        # Process the image
        processed_image = preprocess_image(image_path, args.image_h)
        processed_image = torch.flatten(processed_image)
        
        network.update_beta((step - 1) / (args.max_steps - 1))

        predicted_label = network.act(processed_image, eps)
        next_image, reward, done = env_step(predicted_label, true_label, processed_image)
        
        network.step(processed_image, predicted_label, reward, next_image, done)
        
        scores_window.append(reward)
        scores.append(reward)

        if logger is not None and step % 100 == 0:
            avg_score = np.mean(scores_window)
            logger.log_metrics(step, eps, reward, avg_score)
            torch.save(network.qnetwork_local.state_dict(), 'checkpoint.pth')
            print('\rStep: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(step, eps, reward, avg_score))
        else:
            print('\rStep: {}\tEpsilon: {:.2f}\tScore: {:.2f}\tAverage Score: {:.2f}'.format(step, eps, reward, np.mean(scores_window)), end="")

        if eps > args.eps_end:
            eps -= args.eps_decay
        else:
            eps = args.eps_end

    return scores

def main():
    args = Args()

    if args.wandb: 
        logger = Logger(args)
    else: 
        logger = None

    if not args.load_policy:
        scores = dqn(args, logger)
        rename_file('checkpoint.pth', 'learned_policy.pth')

    test_scores = test_dqn(args, logger)

    if args.wandb: logger.close()

if __name__ == '__main__':
    main()
