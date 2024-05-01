import cv2
import numpy as np
import pandas as pd
import os

images_dir = 'DATA/'
image_files = os.listdir(images_dir)
color_hist_rgb_features_list = []

def convert_to_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_rgb, hist_rgb)
    return hist_rgb.flatten()

# Sử dụng Color Histogram (RGB) để trích xuất đặc trưng cho từng ảnh
dem = 0
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)

    print(convert_to_rgb(img_path).shape)
    # Flatten histogram thành mảng 1 chiều và lưu vào danh sách
    color_hist_rgb_features_list.append({'image_name': img_file, 'color_hist_rgb_features': convert_to_rgb(img_path)})
    
    print(dem, end=' ')
    print("Trích rút thành công!\n")
    dem = dem + 1

    np.save("DACTRUNG/color_hist_rgb.npy", color_hist_rgb_features_list)
