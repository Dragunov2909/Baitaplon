import cv2
from ultralytics import YOLO

# 'yolov8n.pt' là phiên bản "nano" (nhẹ nhất, chạy nhanh nhất cho laptop thường)
model = YOLO('yolov8n.pt')

# 2. Mở Camera 
cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Không thể mở camera")
    exit()

print("Đang chạy demo... Nhấn chữ 'q' để thoát.")

while True:
    # 3. Đọc từng khung hình từ camera
    ret, frame = cap.read()
    if not ret:
        print("Không nhận được khung hình (frame). Đang thoát...")
        break

    # 4. Đưa khung hình cho AI "nhìn" và nhận diện
    
    results = model(frame, stream=True)

    # 5. Xử lý kết quả trả về
    for result in results:
        # Vẽ các khung chữ nhật (bounding boxes) và tên vật thể lên hình
        annotated_frame = result.plot()

        # 6. Hiển thị hình ảnh đã vẽ lên màn hình
        cv2.imshow("AI Nhan dien Dong vat & Nguoi", annotated_frame)

    # 7. Nhấn phím 'q' để thoát chương trình
    if cv2.waitKey(1) == ord('q'):
        break

# Giải phóng camera và đóng cửa sổ
cap.release()
cv2.destroyAllWindows()