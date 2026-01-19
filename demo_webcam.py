import cv2
import numpy as np
import tensorflow as tf
from scipy.spatial import distance as dist
from collections import deque, OrderedDict

# ==========================================
# 1. THE TRACKER CLASSES
# ==========================================
class TrackedSubject:
    def __init__(self, object_id, buffer_size=10):
        self.object_id = object_id
        self.centroid = (0, 0)
        self.box = (0, 0, 0, 0) 
        
        # History of predictions (1.0 = Mask, 0.0 = No Mask)
        self.prediction_history = deque(maxlen=buffer_size)
        self.disappeared_count = 0 

    def update_stats(self, centroid, box, current_mask_prob):
        self.centroid = centroid
        self.box = box
        self.disappeared_count = 0
        self.prediction_history.append(current_mask_prob)

    def get_smoothed_decision(self):
        if not self.prediction_history:
            return "Unknown", (0,0,255), 0.0
            
        # Average probability over the last N frames
        avg_score = sum(self.prediction_history) / len(self.prediction_history)
        
        # Determine Label and Confidence
        if avg_score > 0.5:
            label = "Mask"
            color = (0, 255, 0)  # Green
            confidence = avg_score # e.g., 0.9 = 90%
        else:
            label = "No Mask"
            color = (0, 0, 255)  # Red
            confidence = 1.0 - avg_score # e.g., 0.1 becomes 0.9 (90%)

        return label, color, confidence

class MaskTracker:
    def __init__(self, max_disappeared=10, buffer_size=10):
        self.next_object_id = 0
        self.subjects = OrderedDict()
        self.max_disappeared = max_disappeared
        self.buffer_size = buffer_size

    def register(self, centroid, box, mask_prob):
        subject = TrackedSubject(self.next_object_id, self.buffer_size)
        subject.update_stats(centroid, box, mask_prob)
        self.subjects[self.next_object_id] = subject
        self.next_object_id += 1

    def deregister(self, object_id):
        del self.subjects[object_id]

    def update(self, rects, mask_probs):
        # If no detections, increment disappeared counts
        if len(rects) == 0:
            for object_id in list(self.subjects.keys()):
                self.subjects[object_id].disappeared_count += 1
                if self.subjects[object_id].disappeared_count > self.max_disappeared:
                    self.deregister(object_id)
            return self.subjects

        # Calculate centroids
        input_centroids = np.zeros((len(rects), 2), dtype="int")
        for (i, (x, y, w, h)) in enumerate(rects):
            input_centroids[i] = (int(x + w / 2.0), int(y + h / 2.0))

        # Register new people if we are empty
        if len(self.subjects) == 0:
            for i in range(len(rects)):
                self.register(input_centroids[i], rects[i], mask_probs[i])
        else:
            object_ids = list(self.subjects.keys())
            object_centroids = [t.centroid for t in self.subjects.values()]

            # Match inputs to existing IDs using distance
            D = dist.cdist(np.array(object_centroids), input_centroids)
            rows = D.min(axis=1).argsort()
            cols = D.argmin(axis=1)[rows]

            used_rows = set()
            used_cols = set()

            for (row, col) in zip(rows, cols):
                if row in used_rows or col in used_cols:
                    continue
                
                # Update the matched person
                object_id = object_ids[row]
                self.subjects[object_id].update_stats(input_centroids[col], rects[col], mask_probs[col])

                used_rows.add(row)
                used_cols.add(col)

            # Handle disappeared
            unused_rows = set(range(0, D.shape[0])).difference(used_rows)
            for row in unused_rows:
                object_id = object_ids[row]
                self.subjects[object_id].disappeared_count += 1
                if self.subjects[object_id].disappeared_count > self.max_disappeared:
                    self.deregister(object_id)

            # Handle new faces
            unused_cols = set(range(0, D.shape[1])).difference(used_cols)
            for col in unused_cols:
                self.register(input_centroids[col], rects[col], mask_probs[col])

        return self.subjects
    
# ==========================================
# 2. MAIN LOGIC (CẬP NHẬT ẢNH XÁM)
# ==========================================

# Load Models
# Đảm bảo model h5 này được huấn luyện với đầu vào Gray (1 channel)
model = tf.keras.models.load_model("models/mask_detector_model.h5")
face_detector = cv2.FaceDetectorYN.create(
    model="models/yunet.onnx", config="", input_size=(320, 320),
    score_threshold=0.7, nms_threshold=0.3, top_k=5000
)

tracker = MaskTracker(max_disappeared=10, buffer_size=5)
cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret: break

    h, w = frame.shape[:2]
    face_detector.setInputSize((w, h))

    faces_rects = []
    batch_faces = [] 

    # 1. DETECT FACES
    _, faces = face_detector.detect(frame)

    if faces is not None:
        for face in faces:
            x, y, w_face, h_face = face[:4].astype(int)
            
            # Boundary checks
            x, y = max(0, x), max(0, y)
            face_img = frame[y:y+h_face, x:x+w_face]
            if face_img.size == 0: continue

            # --- PRE-PROCESSING (ẢNH XÁM) ---
            # 1. Chuyển sang ảnh xám
            img = cv2.cvtColor(face_img, cv2.COLOR_BGR2GRAY) 
            # 2. Resize về 128x128
            img = cv2.resize(img, (128, 128)) 
            # 3. Chuẩn hóa pixel về [0, 1]
            img = img.astype("float32") / 255.0 
            # 4. Reshape thêm chiều channel (128, 128, 1) để khớp kiến trúc CNN
            img = np.expand_dims(img, axis=-1) 
            
            batch_faces.append(img)
            faces_rects.append((x, y, w_face, h_face))

    # 2. BATCH PREDICTION
    faces_probs = []
    if len(batch_faces) > 0:
        batch_input = np.array(batch_faces)
        # Tiến hành dự đoán trên batch ảnh xám
        preds = model.predict(batch_input, verbose=0)
        
        for pred in preds:
            faces_probs.append(pred[0])

    # 3. UPDATE TRACKER
    tracked_subjects = tracker.update(faces_rects, faces_probs)

    # 4. DRAW RESULTS
    for object_id, subject in tracked_subjects.items():
        # UPDATED: Now unpacking 3 values (label, color, score)
        label, color, confidence = subject.get_smoothed_decision()
        
        (x, y, w_box, h_box) = subject.box
        
        cv2.rectangle(frame, (x, y), (x+w_box, y+h_box), color, 2)
        
        # Draw ID
        cv2.putText(frame, f"ID: {object_id}", (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, (255, 255, 0), 2)
        
        # UPDATED: Draw Label + Percentage
        # "Mask: 98.5%"
        label_text = f"{label}: {confidence * 100:.2f}%"
        
        cv2.putText(frame, label_text, (x, y + h_box + 25),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.75, color, 2)

    cv2.imshow("Mask Detection (Batch Mode)", frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()