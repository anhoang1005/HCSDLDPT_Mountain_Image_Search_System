import cv2
import numpy as np
import pandas as pd
import os

images_dir = 'DATA/'
image_files = os.listdir(images_dir)
color_hist_rgb_features_list = []

def convert_to_red_rgb(img_path):
    image = cv2.imread(img_path)
    hist_red = cv2.calcHist([image], [0], None, [16], [0, 256])
    #cv2.normalize(hist_red, hist_red)
    return hist_red.flatten()

def convert_to_green_rgb(img_path):
    image = cv2.imread(img_path)
    hist_green = cv2.calcHist([image], [1], None, [16], [0, 256])
    return hist_green.flatten()

def convert_to_blue_rgb(img_path):
    image = cv2.imread(img_path)
    hist_blue = cv2.calcHist([image], [2], None, [16], [0, 256])
    return hist_blue.flatten()

def convert_to_rgb(img_path):
    image = cv2.imread(img_path)
    hist_red = cv2.calcHist([image], [0], None, [16], [0, 256])
    hist_green = cv2.calcHist([image], [1], None, [16], [0, 256])
    hist_blue = cv2.calcHist([image], [2], None, [16], [0, 256])
    #cv2.normalize(hist_red, hist_red)
    # Concatenate các mảng vào một tuple trước khi truyền vào np.concatenate
    hist_rgb = np.concatenate((hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()))
    return hist_rgb

# Sử dụng Color Histogram (RGB) để trích xuất đặc trưng cho từng ảnh
dem = 0
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)

    print(convert_to_rgb(img_path).shape)
    # print(convert_to_red_rgb(img_path).shape)
    # print(convert_to_green_rgb(img_path).shape)
    # print(convert_to_blue_rgb(img_path).shape)
    # Flatten histogram thành mảng 1 chiều và lưu vào danh sách
    #color_hist_rgb_features_list.append({'image_name': img_file, 'red_features': convert_to_green_rgb(img_path), 'green_features': convert_to_blue_rgb(img_path), 'blue_features': convert_to_red_rgb(img_path)})
    color_hist_rgb_features_list.append({'image_name': img_file, 'rgb_features': convert_to_rgb(img_path)})

    print(dem, end=' ')
    print("Trích rút thành công!\n")
    dem = dem + 1

    np.save("DACTRUNG/color_hist_r_g_b.npy", color_hist_rgb_features_list)
