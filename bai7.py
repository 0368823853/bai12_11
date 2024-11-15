import torch
import torchvision
from PIL import Image
from torchvision import transforms
import matplotlib.pyplot as plt
import matplotlib.patches as patches
import time

# 1. Tải mô hình Faster R-CNN đã được huấn luyện sẵn trên COCO dataset
model = torchvision.models.detection.fasterrcnn_resnet50_fpn(pretrained=True)
model.eval()  # Đặt mô hình ở chế độ đánh giá


# 2. Định nghĩa hàm tiền xử lý ảnh đầu vào
def preprocess_image(image_path):
    img = Image.open(image_path).convert("RGB")
    transform = transforms.Compose([transforms.ToTensor()])
    return transform(img).unsqueeze(0)  # Thêm batch dimension


# 3. Đưa ảnh vào mô hình và nhận kết quả dự đoán
def predict(model, image_tensor, threshold=0.5):
    with torch.no_grad():
        predictions = model(image_tensor)
    return predictions[0]


# 4. Vẽ bounding box và nhãn đối tượng lên ảnh
def plot_predictions(image_path, predictions, threshold=0.5):
    img = Image.open(image_path)
    fig, ax = plt.subplots(1, figsize=(12, 9))
    ax.imshow(img)

    for idx, (box, score, label) in enumerate(zip(predictions['boxes'], predictions['scores'], predictions['labels'])):
        if score >= threshold:
            # Vẽ bounding box
            x_min, y_min, x_max, y_max = box
            rect = patches.Rectangle((x_min, y_min), x_max - x_min, y_max - y_min,
                                     linewidth=2, edgecolor='r', facecolor='none')
            ax.add_patch(rect)

            # Thêm nhãn và độ tin cậy
            label_text = f"{label.item()} ({score:.2f})"
            ax.text(x_min, y_min, label_text, color='white', fontsize=12,
                    bbox=dict(facecolor='red', edgecolor='none', pad=0.5))

    plt.axis('off')
    plt.show()


# 5. Tính toán thời gian suy luận trung bình
def calculate_inference_time(model, images):
    times = []
    for image in images:
        image_tensor = preprocess_image(image)

        start = time.time()
        predict(model, image_tensor)
        end = time.time()

        times.append(end - start)

    return sum(times) / len(times)


# 6. Thực hiện nhận diện đối tượng và hiển thị kết quả
def detect_and_display(image_path, threshold=0.5):
    image_tensor = preprocess_image(image_path)
    predictions = predict(model, image_tensor)
    plot_predictions(image_path, predictions, threshold)


# Ví dụ sử dụng
image_path = "images/avt.jpg"  # Đường dẫn đến ảnh cần kiểm thử
detect_and_display(image_path, threshold=0.5)

# Tính thời gian suy luận trung bình trên một tập ảnh kiểm thử
test_images = ["vetinh1.jpg", "vetinh2.jpg", "vetinh3.jpg"]  # Đường dẫn tới các ảnh kiểm thử
avg_inference_time = calculate_inference_time(model, test_images)
print(f"Thời gian suy luận trung bình: {avg_inference_time:.4f} giây")
