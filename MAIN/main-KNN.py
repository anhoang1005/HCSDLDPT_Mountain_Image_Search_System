import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import pandas as pd
import os
import matplotlib.pyplot as plt

# Thư mục chứa các ảnh
images_dir = 'DATA/'
# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

def trich_rut_dac_trung_hog(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (fd, hog_image) = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L2')
    return fd

def trich_rut_dac_trung_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    cv2.normalize(hist_rgb, hist_rgb)
    return hist_rgb.flatten()

def convert_to_rgb(img_path):
    image = cv2.imread(img_path)
    hist_red = cv2.calcHist([image], [0], None, [16], [0, 256])
    hist_green = cv2.calcHist([image], [1], None, [16], [0, 256])
    hist_blue = cv2.calcHist([image], [2], None, [16], [0, 256])
    #cv2.normalize(hist_red, hist_red)
    hist_rgb = np.concatenate((hist_red.flatten(), hist_green.flatten(), hist_blue.flatten()))
    return hist_rgb

def trich_rut_dac_trung_hog_rgb(img_path):
    image = cv2.imread(img_path)
    hist_rgb = cv2.calcHist([image], [0, 1, 2], None, [8, 8, 8], [0, 256, 0, 256, 0, 256])
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    fd, hog_image = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L2')
    cv2.normalize(hist_rgb, hist_rgb)
    # Kết hợp đặc trưng HOG và Color Histogram và lưu vào danh sách
    combined_features = np.concatenate((fd, hist_rgb.flatten()))
    return combined_features

def khoangcach_euclidean(x, y):
    #Nếu kích thước vector không giống nhau in ra None
    if len(x) != len(y):
        return None
    #Tính khoảng cách euclidean (bản chất là trừ vector)
    squared_distance = 0
    for i in range(len(x)):
        squared_distance += (x[i] - y[i]) ** 2
    return squared_distance ** 0.5

def chuan_hoa_euclidean_ve_khoang(distance):
    #Chuẩn hóa khoảng cách euclidean về đoạn [0,1] để dễ tính toán (dùng công thức Min-Max)
    max_distance = np.max(distance)
    min_distance = np.min(distance)
    chuanhoa_distances = (distance-min_distance)/(max_distance - min_distance)
    return chuanhoa_distances

def danhsach_euclidean(input_image_dt, dt_features_list):
    #Hàm chạy tính khoảng cách euclidean của input với tất cả ảnh trong data
    distance_euclidean = np.array([])
    for i, dt_data in enumerate(dt_features_list):
        distance_euclidean = np.append(distance_euclidean, khoangcach_euclidean(dt_data, input_image_dt))
    distance_euclidean = chuan_hoa_euclidean_ve_khoang(distance_euclidean)
    return distance_euclidean

def knn(X_train, Y_train, X_new, k):
    # Tính khoảng cách Euclidean giữa input và các điểm trong tập huấn luyện
    hog_distances_euclidean = danhsach_euclidean(X_new[0], X_train[0])
    rgb_distances_euclidean = danhsach_euclidean(X_new[1], X_train[1])
    #hog_rgb_distances_euclidean = danhsach_euclidean(X_new[2], X_train[2])

    # Kết hợp khoảng cách từ HOG và RGB features
    hog_rgb_distances = (hog_distances_euclidean + rgb_distances_euclidean)/2
    # Sắp xếp các điểm theo khoảng cách tăng dần
    sorted_similarities_euclidean = np.argsort(hog_rgb_distances)[::1]

    # Chọn k điểm gần nhất
    top_similar_images = []
    for idx in sorted_similarities_euclidean[:k]:
        image_name = Y_train[0][idx]
        similarity = hog_rgb_distances[idx]
        top_similar_images.append((image_name, similarity))
    return top_similar_images

#Đương dẫn của file lưu các bộ đặc trưng
path_hog ="DACTRUNG/hog.npy"
path_rgb ="DACTRUNG/color_hist_rgb.npy"
path_hog_rgb ="DACTRUNG/hog_rgb_features.npy"

data_hog = np.load(path_hog , allow_pickle="True")
data_rgb = np.load(path_rgb , allow_pickle="True")
data_hog_rgb = np.load(path_hog_rgb , allow_pickle="True")

#Sắp xếp các đặc trưng vào danh sách riêng
data_name_list = [item['image_name'] for item in data_hog] 
hog_features_list = [item['hog_features'] for item in data_hog] 
rgb_features_list = [item['color_hist_rgb_features'] for item in data_rgb]
hog_rgb_features_list = [item['hog_rgb_features'] for item in data_hog_rgb]

#Đừờng dẫn ănh input
input_path = 'DATA/anh1.png'
#Trích rút các đặc trưng của ảnh input
input_image_hog_features = trich_rut_dac_trung_hog(input_path)
input_image_rgb_features = trich_rut_dac_trung_rgb(input_path)
input_image_hog_rgb_features = trich_rut_dac_trung_hog_rgb(input_path)

#Đưa đặc trưng vào tập train và test để chuẩn bị chạy
X_new = [input_image_hog_features, input_image_rgb_features, input_image_hog_rgb_features]
X_train = [hog_features_list, rgb_features_list, hog_rgb_features_list]
Y_train = [data_name_list]

# Lấy ra danh sách ảnh giống nhất
k = 4
top_similar_images = knn(X_train, Y_train, X_new, k)

# In ra thông tin tat ca cac anh
for idx, (img_file, dissimilarity) in enumerate(top_similar_images):
    print(f"STT: {idx} - {img_file} - Độ sai khác: {dissimilarity}")
print('\n')

input_image = cv2.imread(input_path)
input_rgb = cv2.cvtColor(input_image, cv2.COLOR_BGR2RGB)
plt.subplot(3, 2, 1)
plt.imshow(input_rgb)
plt.axis('off')
plt.title('Input Image')

# Vẽ ảnh khác biệt nhất từ kết hợp
for idx, (img_file, similarity_combined) in enumerate(top_similar_images):
    img_path = os.path.join(images_dir, img_file)
    image = cv2.imread(img_path)
    image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        
    plt.subplot(3, 2, idx+3)  # Vẽ ảnh trên 1 dòng, 3 cột
    plt.imshow(image_rgb)
    plt.axis('off')  # Tắt trục tọa độ
    plt.title(f'{img_file}\n loss: {similarity_combined:.8f}')  # Hiển thị độ tuong dong

plt.tight_layout()  # Đảm bảo layout hợp lý
plt.show()