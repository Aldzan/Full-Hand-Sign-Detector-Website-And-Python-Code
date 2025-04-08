from flask import Flask, render_template, Response
import cv2
from baru import HandSignDetector  # Import dari file new dt.py

app = Flask(__name__)

# Inisialisasi detektor tangan
detector = HandSignDetector()

def generate_frames():
    """Mengambil frame dari kamera dan menjalankan deteksi tangan."""
    cap = cv2.VideoCapture(0)
    while True:
        success, frame = cap.read()
        if not success:
            break

        # Flip frame agar tidak terbalik
        frame = cv2.flip(frame, 1)

        # Jalankan deteksi tangan
        frame, landmarks_list, bounding_boxes = detector.extract_hand_landmarks(frame)

        # Gambar kotak di sekitar tangan dan tampilkan hasil deteksi
        for i, (min_x, min_y, max_x, max_y) in enumerate(bounding_boxes):
            cv2.rectangle(frame, (min_x, min_y), (max_x, max_y), (0, 255, 0), 2)

            # Jika ada landmarks, jalankan prediksi
            if landmarks_list:
                sign_name = detector.detect_sign(landmarks_list[i])  # Prediksi tanda tangan

                # Tampilkan teks hasil deteksi di atas bounding box
                cv2.putText(frame, sign_name, (min_x, min_y - 10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 0, 255), 2)

        # Konversi frame ke format JPEG
        _, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

    cap.release()
@app.route('/')
def index():
    """Menampilkan halaman utama."""
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    """Mengirimkan video streaming ke frontend."""
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == '__main__':
    app.run(debug=True)