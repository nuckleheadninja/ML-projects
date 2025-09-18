import os
import shutil
import random

# Paths
data_dir = r"C:\Users\User\.cache\kagglehub\datasets\lukex9442\indian-bovine-breeds\versions\1\Indian_bovine_breeds"
output_base = r"C:\Users\User\indian_bovine_split"

train_dir = os.path.join(output_base, "train")
val_dir   = os.path.join(output_base, "val")
test_dir  = os.path.join(output_base, "test")

# Clean old split if exists
for d in [train_dir, val_dir, test_dir]:
    if os.path.exists(d):
        shutil.rmtree(d)
    os.makedirs(d)

def split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15):
    assert abs(train_ratio + val_ratio + test_ratio - 1.0) < 1e-6, "Ratios must sum to 1!"

    for class_name in os.listdir(data_dir):
        class_path = os.path.join(data_dir, class_name)
        if not os.path.isdir(class_path):
            continue

        # filter only image files
        images = [f for f in os.listdir(class_path) if f.lower().endswith((".jpg", ".jpeg", ".png"))]
        random.shuffle(images)

        n_total = len(images)
        n_train = int(train_ratio * n_total)
        n_val   = int(val_ratio * n_total)

        train_imgs = images[:n_train]
        val_imgs   = images[n_train:n_train+n_val]
        test_imgs  = images[n_train+n_val:]

        for folder, img_list in [(train_dir, train_imgs), (val_dir, val_imgs), (test_dir, test_imgs)]:
            os.makedirs(os.path.join(folder, class_name), exist_ok=True)
            for img in img_list:
                shutil.copy(os.path.join(class_path, img), os.path.join(folder, class_name, img))

        print(f"✅ {class_name}: {len(train_imgs)} train, {len(val_imgs)} val, {len(test_imgs)} test")

    print("\n🎯 Dataset splitting completed (all data kept)!")
    print("📂 Train:", train_dir)
    print("📂 Val:", val_dir)
    print("📂 Test:", test_dir)

split_dataset(train_ratio=0.7, val_ratio=0.15, test_ratio=0.15)
