import os
import shutil
import random

# Path to your dataset
data_dir = r"C:\Users\User\.cache\kagglehub\datasets\lukex9442\indian-bovine-breeds\versions\1\Indian_bovine_breeds"

# Output base folder for splitted data
output_base = r"C:\Users\User\indian_bovine_split"

# Train, Validation, Test folders
train_dir = os.path.join(output_base, "train")
val_dir = os.path.join(output_base, "val")
test_dir = os.path.join(output_base, "test")

# Create base folders
for d in [train_dir, val_dir, test_dir]:
    os.makedirs(d, exist_ok=True)

def balance_and_split_dataset():
    class_counts = {}
    
    # Step 1: Count images in each class
    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if os.path.isdir(class_path):
            class_counts[class_name] = len(os.listdir(class_path))

    # Find minimum count across all classes
    min_count = min(class_counts.values())
    print(f"⚖️ Balancing dataset: each class will have {min_count} images.")

    # Step 2: Process each class
    for class_name, count in class_counts.items():
        class_path = os.path.join(data_dir, class_name)
        images = os.listdir(class_path)
        random.shuffle(images)

        # Select only min_count images for balancing
        balanced_images = images[:min_count]

        # Split 30% test, remaining 70% -> 80% train, 20% val
        n_test = int(0.1 * min_count)
        test_imgs = balanced_images[:n_test]

        remaining = balanced_images[n_test:]
        random.shuffle(remaining)

        n_train = int(0.7 * min_count)
        train_imgs = remaining[:n_train]
        val_imgs = remaining[n_train:]

        # Create subfolders for each class
        os.makedirs(os.path.join(train_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(val_dir, class_name), exist_ok=True)
        os.makedirs(os.path.join(test_dir, class_name), exist_ok=True)

        # Copy Train images
        for img in train_imgs:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(train_dir, class_name, img))

        # Copy Validation images
        for img in val_imgs:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(val_dir, class_name, img))

        # Copy Test images
        for img in test_imgs:
            shutil.copy(os.path.join(class_path, img),
                        os.path.join(test_dir, class_name, img))

        print(f"✅ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\n🎯 Dataset balancing + splitting completed!")
    print("📂 Train folder saved at: ", train_dir)
    print("📂 Validation folder saved at: ", val_dir)
    print("📂 Test folder saved at: ", test_dir)

balance_and_split_dataset()
