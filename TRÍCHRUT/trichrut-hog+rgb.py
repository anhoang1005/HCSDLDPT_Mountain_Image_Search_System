import cv2
import numpy as np
import pandas as pd
import os
from skimage.feature import hog

# Thư mục chứa các ảnh
images_dir = 'DATA/'
# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)
# Danh sách chứa đặc trưng HOG và Color Histogram (HSV) và định danh của từng ảnh
combined_features_list = []

def convert_to_hog_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L1')
    cv2.normalize(hist_rgb, hist_rgb)
    # Kết hợp đặc trưng HOG và Color Histogram và lưu vào danh sách
    combined_features = np.concatenate((fd, hist_rgb.flatten()))
    print(combined_features.shape)
    return combined_features
    

# Sử dụng HOG và Color Histogram (HSV) để trích xuất đặc trưng cho từng ảnh
dem = 0
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    combined_features_list.append({'image_name': img_file, 'hog_rgb_features': convert_to_hog_rgb(img_path)})
    print(dem, end=' ')
    print("Trích rút thành công!\n")
    dem = dem + 1

np.save("DACTRUNG/hog_rgb_features.npy", combined_features_list)
