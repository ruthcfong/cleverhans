import numpy as np

from torchvision import transforms
from PIL import Image


def crop_image(img_path, out_path, size=(299, 299)):
    transform = transforms.Compose([
        transforms.Scale(np.max(size)),
        transforms.CenterCrop(size)])
    img = Image.open(img_path)
    img_ = transform(img)
    img_.save(out_path)


def main():
    img_path = 'test_image.JPEG'
    out_path = 'cropped_img.jpeg'
    crop_image(img_path, out_path, (299, 299))


if __name__ == '__main__':
    main()
