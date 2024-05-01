import cv2
import numpy as np
from skimage.feature import hog
from skimage import exposure
import pandas as pd
import os

from sklearn.cluster import KMeans

# Thư mục chứa các ảnh
images_dir = 'DATA/'

# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

# Danh sách chứa đặc trưng HOG và định danh của từng ảnh
hog_features_list = []

def convert_to_hog(img_path):
    image = cv2.imread(img_path)
    gray_image = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    (fd, hog_image) = hog(gray_image, orientations=9, pixels_per_cell=(8, 8), cells_per_block=(1, 1), visualize=True, block_norm='L1')
    return fd

# Thực hiện trích xuất đặc trưng HOG cho từng ảnh
list_hog_data = []
for i,img_file in enumerate(image_files): 
    img_path = os.path.join(images_dir, img_file)
    list_hog_data.append(convert_to_hog(img_path))
    hog_features_list.append({'image_name': img_file, 'hog_features': convert_to_hog(img_path)})
    print(f'{i} Trich rut thanh cong!', end='\n')
print('\n')

np.save("DACTRUNG/hog.npy", hog_features_list)

#print(list_hog_data)

# # Chuyển danh sách đặc trưng HOG thành DataFrame
# hog_features_df = pd.DataFrame(hog_features_list)
# print(hog_features_df['hog_features'])

# input_path = 'TEST/test1.png'
# input_image_features = convert_to_hog(input_path)
# print(input_image_features.shape)

# # Chuẩn bị dữ liệu cho việc huấn luyện K-Means
# X_train = np.array(list_hog_data)

# # Sử dụng K-Means để phân cụm các đặc trưng HOG của các ảnh
# kmeans = KMeans(n_clusters=3, random_state=0)
# clusters = kmeans.fit_predict(X_train)

# # Dự đoán nhãn cụm của ảnh input
# input_cluster = kmeans.predict([input_image_features])

# # Tìm các ảnh trong cùng cụm với ảnh input
# similar_images_indices = np.where(clusters == input_cluster)[0]

# # Lấy ra 3 ảnh giống nhất
# top_similar_images = []
# for idx in similar_images_indices:
#     image_name = hog_features_list.loc[idx, 'image_name']
#     top_similar_images.append(image_name)

# print(f"Top 3 ảnh giống nhất với ảnh input: {top_similar_images[:3]}")



