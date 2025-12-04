import cv2
import mediapipe as mp
from ultralytics import YOLO

# 1. Khởi tạo YOLO (Nhận diện vật thể)
print("Đang tải mô hình YOLO...")
yolo_model = YOLO('yolov8n.pt')

# 2. Khởi tạo MediaPipe (Nhận diện bàn tay)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(
    max_num_hands=1,             # Chỉ nhận diện 1 bàn tay để tránh rối
    min_detection_confidence=0.7 # Độ tin cậy tối thiểu 70%
)
mp_draw = mp.solutions.drawing_utils # Công cụ để vẽ các dấu chấm

# 3. Mở Camera
cap = cv2.VideoCapture(0)

# Hàm đếm số ngón tay đang giơ lên
def count_fingers(hand_landmarks):
    # Danh sách các điểm đầu ngón tay (Tip) trong MediaPipe:
    # 4: Ngón cái, 8: Trỏ, 12: Giữa, 16: Áp út, 20: Út
    finger_tips = [8, 12, 16, 20] 
    count = 0
    
    # Lấy danh sách toạ độ các điểm
    lm_list = hand_landmarks.landmark

    # --- Xử lý 4 ngón dài (Trừ ngón cái) ---
    # Logic: Nếu đầu ngón tay (Tip) cao hơn khớp nối (Pip) thì là đang giơ
    # Lưu ý: Trong ảnh, trục Y càng cao thì giá trị càng NHỎ (gốc 0,0 ở góc trên trái)
    for tip in finger_tips:
        if lm_list[tip].y < lm_list[tip - 2].y:
            count += 1

    # --- Xử lý ngón cái (Hơi khác biệt vì nó chuyển động ngang) ---
    # Nếu đầu ngón cái nằm xa hơn khớp nối theo trục X
    if lm_list[4].x > lm_list[3].x: # Logic này đúng cho tay phải, tay trái sẽ ngược lại
        count += 1
        
    return count

print("Chương trình bắt đầu! Giơ tay lên camera...")

while True:
    ret, frame = cap.read()
    if not ret: break

    # Lật ngược ảnh cho giống gương (tùy chọn)
    frame = cv2.flip(frame, 1)
    
    # MediaPipe cần ảnh màu RGB, còn OpenCV dùng BGR
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    
    # Xử lý tìm bàn tay
    result_hands = hands.process(rgb_frame)
    
    finger_count = 0 # Mặc định là 0 ngón

    # Nếu tìm thấy bàn tay
    if result_hands.multi_hand_landmarks:
        for hand_lms in result_hands.multi_hand_landmarks:
            # A. Vẽ các dấu chấm và đường nối lên tay
            mp_draw.draw_landmarks(frame, hand_lms, mp_hands.HAND_CONNECTIONS)
            
            # B. Đếm số ngón tay
            finger_count = count_fingers(hand_lms)

    # --- XỬ LÝ LỆNH THEO SỐ NGÓN TAY ---
    
    # Lệnh 1: In chữ "deptrai"
    if finger_count == 1:
        cv2.putText(frame, "DEP TRAI QUA!", (50, 100), 
                    cv2.FONT_HERSHEY_SIMPLEX, 2, (255, 0, 0), 5)
        
    # Lệnh 2: Nhận diện vật thể (Chạy YOLO)
    elif finger_count == 2:
        # Đưa khung hình vào YOLO
        yolo_results = yolo_model(frame, stream=True, verbose=False)
        
        # Vẽ kết quả YOLO lên khung hình
        for r in yolo_results:
            boxes = r.boxes
            for box in boxes:
                # Lấy tọa độ khung
                x1, y1, x2, y2 = box.xyxy[0]
                x1, y1, x2, y2 = int(x1), int(y1), int(x2), int(y2)
                
                # Lấy tên vật thể
                cls = int(box.cls[0])
                name = yolo_model.names[cls]
                
                # Vẽ khung và tên
                cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 0, 255), 2)
                cv2.putText(frame, name, (x1, y1 - 10), 
                            cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 0, 255), 2)
        
        cv2.putText(frame, "Mode: OBJECT DETECTION", (50, 50), 
                    cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 0, 255), 2)

    # Lệnh 3: Tắt chương trình
    elif finger_count == 3:
        print("Đã nhận lệnh tắt chương trình!")
        break
    
    # Hiển thị số ngón tay lên màn hình để dễ debug
    cv2.putText(frame, f"Ngon tay: {finger_count}", (10, 450), 
                cv2.FONT_HERSHEY_PLAIN, 2, (255, 0, 0), 2)

    cv2.imshow("Dieu khien bang tay", frame)
    
    if cv2.waitKey(1) == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()