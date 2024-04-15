def get_random_image():
    # Get a random image from 'images' directory
    pass

def preprocess_image(image):
    # Standardize the iamge format
    pass

def calculate_eps_decay(eps_start, eps_end, n_steps, eps_percentage):
    # Calculate the rate epsilon should decay
    effective_steps = n_steps * eps_percentage
    decrement_per_step = (eps_start - eps_end) / effective_steps
    return decrement_per_step