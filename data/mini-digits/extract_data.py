import numpy as np
import os
import pickle

with open(os.path.join(data_dir, "tiny-handwritten-as-rotation-angles.pkl"), 'rb') as f:
    mnist = pickle.load(f)
    train_imgs, train_labels, test_imgs, test_labels = mnist["training_images"], mnist["training_labels"], mnist[
        "test_images"], mnist["test_labels"]

# save the individual images
counter = 0
for i in range(len(train_imgs)):
    label = train_labels[i]
    img_array = train_imgs[i] / (2 * np.pi) * 255
    img = Image.fromarray(img_array.astype('uint8'))
    save_name = f"image_train_{counter:04}_label_{label}.png"

    img.save(os.path.join(image_png_dir, save_name))
    counter += 1

for i in range(len(test_imgs)):
    label = test_labels[i]
    img_array = test_imgs[i] / (2 * np.pi) * 255
    img = Image.fromarray(img_array.astype('uint8'))
    save_name = f"image_test_{counter:04}_label_{label}.png"

    img.save(os.path.join(image_png_dir, save_name))
    counter += 1