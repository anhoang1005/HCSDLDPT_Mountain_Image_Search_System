import os

# Thư mục chứa các ảnh
images_dir = 'DATA1/'

# Danh sách tên các file ảnh
image_files = os.listdir(images_dir)

# Lặp qua từng file ảnh và đổi tên
for i, img_file in enumerate(image_files):
    # Xác định đường dẫn tới file ảnh cũ
    old_path = os.path.join(images_dir, img_file)
    
    # Xác định tên mới cho file ảnh
    new_name = f'anh{i+1}.png'  # Đổi phần mở rộng tùy theo định dạng của ảnh
    
    # Xác định đường dẫn mới cho file ảnh
    new_path = os.path.join(images_dir, new_name)
    
    # Đổi tên file ảnh
    os.rename(old_path, new_path)
    print('doi ten anh thanh cong')

print("Đã đổi tên các file ảnh thành công!")
