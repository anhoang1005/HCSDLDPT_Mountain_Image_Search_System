import cv2
import numpy as np
import pandas as pd
import os

# Thư mục chứa các ảnh
images_dir = 'DATA/'

# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

# Danh sách chứa đặc trưng Color Histogram (HSV) và định danh của từng ảnh
color_hist_features_list = []

def convert_to_hsv(img_path):
    image = cv2.imread(img_path)
    hsv_image = cv2.cvtColor(image, cv2.COLOR_BGR2HSV)
    # Tính toán histogram màu sắc
    hist_hsv = cv2.calcHist([hsv_image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    # Chuẩn hóa histogram
    cv2.normalize(hist_hsv, hist_hsv)
    return hist_hsv.flatten()

# Sử dụng Color Histogram (HSV) để trích xuất đặc trưng cho từng ảnh
dem = 0
for img_file in image_files:
    img_path = os.path.join(images_dir, img_file)
    
    # Flatten histogram thành mảng 1 chiều và lưu vào danh sách
    color_hist_features_list.append({'image_name': img_file, 'color_hist_hsv_features': convert_to_hsv(img_path)})
    
    print(dem, end=' ')
    print("Trích rút thành công!\n")
    dem = dem + 1

# # Chuyển đổi danh sách đặc trưng thành DataFrame
# color_hist_features_df = pd.DataFrame(color_hist_features_list)

# # Lưu đặc trưng và định danh vào tệp CSV
# color_hist_features_df.to_csv('DACTRUNG/color_histogram_hsv.csv', index=False)

np.save("DACTRUNG/color_hist_hsv.npy", color_hist_features_list)
