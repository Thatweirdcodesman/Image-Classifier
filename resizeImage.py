import io
from tkinter import Tk, Canvas, NW
import cv2
from PIL import Image, ImageTk
from numpy import asarray


def resize_image(img, max_width=320, max_height=320):
    background = Image.new('RGB', (max_width, max_height), (255, 255, 255))
    # img.show()
    bg_w, bg_h = background.size
    # print(bg_w,bg_h)
    im = Image.fromarray(img)
    width, height = im.size
    ratio = float(height) / float(width)
    new_height = max_height if ratio > 1.0 else int(float(max_width) * ratio)
    new_width = int(float(max_height) / ratio) if ratio > 1.0 else max_width

    im = im.resize((new_width, new_height), Image.ANTIALIAS)
    # im.show()
    width, height = im.size
    offset = ((bg_w - width) // 2, (bg_h - height) // 2)
    background.paste(im, offset)
    # background.show()
    extension = 'png'
    # if '.' in fspath:
    #     extension = fspath.split('.')[-1]
    background.format = 'JPEG' if extension.lower() in ('jpg', 'jpe', 'jpeg', 'jfif') else extension.upper()
    grey_img = background.convert('L')
    resizedImage = asarray(grey_img)
    print("resized Shape: ", resizedImage.shape)
    return resizedImage
# resize_image('/home/hariharan/Documents/Image Classifier Test/edge_eight.jpeg')
