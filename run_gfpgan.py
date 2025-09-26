import os
import cv2
from gfpgan import GFPGANer
from matplotlib import pyplot as plt

# --- Chọn phiên bản model ---
# Các lựa chọn: "1", "1.3", "1.4", "clean"
model_version = "1.4"

if model_version == "clean":
    weight_path = "./gfpgan/weights/GFPGANcleanv1-NoCE-C2.pth"
else:
    weight_path = f"./gfpgan/weights/GFPGANv{model_version}.pth"

# --- Load model ---
restorer = GFPGANer(
    model_path=weight_path,
    upscale=2,                # phóng to 2x
    arch="clean",             # dùng 'clean' cho v1.4
    channel_multiplier=2,
    bg_upsampler=None
)
print(f"✅ Loaded GFPGAN v{model_version} từ {weight_path}")

# --- Thư mục input/output ---
input_dir = "./inputs/whole_imgs"
output_dir = "./results_vscode"
os.makedirs(output_dir, exist_ok=True)

# --- Chạy trên toàn bộ ảnh trong input ---
for img_name in os.listdir(input_dir):
    img_path = os.path.join(input_dir, img_name)
    print("🔄 Đang xử lý:", img_path)

    # Đọc ảnh
    img = cv2.imread(img_path, cv2.IMREAD_COLOR)

    # Phục hồi
    _, _, restored_img = restorer.enhance(
        img, has_aligned=False, only_center_face=False, paste_back=True
    )

    # Lưu kết quả
    save_path = os.path.join(output_dir, f"restored_{img_name}")
    cv2.imwrite(save_path, restored_img)

    print("💾 Đã lưu:", save_path)

print("\n🎉 Hoàn thành! Kết quả nằm trong thư mục:", output_dir)
