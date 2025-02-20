import os

def count_images_in_dirs(base_path):
    # Dictionary to store counts for each directory
    counts = {}
    
    # Walk through the Filtered directory
    filtered_path = os.path.join(base_path, 'Filtered')
    if os.path.exists(filtered_path):
        for dir_name in os.listdir(filtered_path):
            dir_path = os.path.join(filtered_path, dir_name)
            if os.path.isdir(dir_path):
                # Count PNG files in directory
                image_count = len([f for f in os.listdir(dir_path) if f.endswith('.png')])
                counts[dir_name] = image_count
    
    # Print results
    print("\nImage counts in each directory:")
    total = 0
    for dir_name, count in counts.items():
        print(f"{dir_name}: {count} images")
        total += count
    print(f"\nTotal images: {total}")

if __name__ == "__main__":
    base_path = "TestingData"
    count_images_in_dirs(base_path)