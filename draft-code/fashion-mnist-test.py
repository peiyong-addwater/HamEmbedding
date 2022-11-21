import pennylane as qml
import pennylane.numpy as np




def load_mnist(path, kind='train'):
    # from https://github.com/zalandoresearch/fashion-mnist/blob/master/utils/mnist_reader.py
    import os
    import gzip
    import numpy as np

    """Load MNIST data from `path`"""
    labels_path = os.path.join(path,
                               '%s-labels-idx1-ubyte.gz'
                               % kind)
    images_path = os.path.join(path,
                               '%s-images-idx3-ubyte.gz'
                               % kind)

    with gzip.open(labels_path, 'rb') as lbpath:
        labels = np.frombuffer(lbpath.read(), dtype=np.uint8,
                               offset=8)

    with gzip.open(images_path, 'rb') as imgpath:
        images = np.frombuffer(imgpath.read(), dtype=np.uint8,
                               offset=16).reshape(len(labels), 784)

    return images, labels

if __name__ == '__main__':
    from PIL import Image
    data_folder = "/home/peiyongw/Desktop/Research/QML-ImageClassification/data/fashion"
    im, labels = load_mnist(data_folder)
    test_index = 1234
    im_test = Image.fromarray(im[test_index].reshape(28,28))
    print(im_test)
    im_test_label = labels[test_index]
    im_test.save(f"test-image-label-{im_test_label}.pdf")