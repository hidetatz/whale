import gzip
import os
import urllib.request

from tensor import Tensor

cache_dir = os.path.join(os.path.expanduser("~"), ".whale")


def download(url):
    file_name = url[url.rfind("/") + 1 :]
    file_path = os.path.join(cache_dir, file_name)

    if not os.path.exists(cache_dir):
        os.mkdir(cache_dir)

    if os.path.exists(file_path):
        return file_path

    print("Downloading: " + file_name)
    try:
        urllib.request.urlretrieve(url, file_path)
    except (Exception, KeyboardInterrupt) as e:
        if os.path.exists(file_path):
            os.remove(file_path)
        raise
    print("Done")

    return file_path


def print_mnist(pixels, height, width):
    for i in range(height):
        row = ""
        for j in range(width):
            pixel_val = pixels[i * width + j]
            if pixel_val > 0.5:
                row += "██"
            elif pixel_val > 0.3:
                row += "▓▓"
            elif pixel_val > 0.1:
                row += "░░"
            else:
                row += "  "
        print(row)


def mnist():
    url = "https://ossci-datasets.s3.amazonaws.com/mnist"
    train_images_path = download(f"{url}/train-images-idx3-ubyte.gz")
    train_labels_path = download(f"{url}/train-labels-idx1-ubyte.gz")
    test_images_path = download(f"{url}/t10k-images-idx3-ubyte.gz")
    test_labels_path = download(f"{url}/t10k-labels-idx1-ubyte.gz")

    def load_images(path):
        with gzip.open(path, "rb") as f:
            assert int.from_bytes(f.read(4)) == 2051, "magic number is not 2051"
            cnt = int.from_bytes(f.read(4))  # 60000
            h = int.from_bytes(f.read(4))  # 28
            w = int.from_bytes(f.read(4))  # 28

            imgs = []
            for i in range(cnt):
                img = f.read(h * w)
                imgs.append([byte / 255.0 for byte in img])

            return imgs

    def load_labels(path):
        with gzip.open(path, "rb") as f:
            assert int.from_bytes(f.read(4)) == 2049, "magic number is not 2049"
            cnt = int.from_bytes(f.read(4))  # 60000

            lbls = []
            for i in range(cnt):
                lbl = int.from_bytes(f.read(1))
                # onehot = [0, 0, 0, 0, 0, 0, 0, 0, 0, 0]
                # onehot[lbl] = 1
                # lbls.append(onehot)
                lbls.append(lbl)
            return lbls

    train_images = load_images(train_images_path)
    train_labels = load_labels(train_labels_path)
    test_images = load_images(test_images_path)
    test_labels = load_labels(test_labels_path)

    return (train_images, train_labels, test_images, test_labels)


if __name__ == "__main__":
    mnist()
