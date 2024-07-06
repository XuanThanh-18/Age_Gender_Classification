import cv2
import tkinter as tk
from tkinter import filedialog
from threading import Thread
from PIL import Image,ImageTk
import os
import numpy as np
import matplotlib.pyplot as plt
base_path = r'C:\Users\hp\Downloads\age-gender-detector-python-master\age-gender-detector-python-master'
def getFaceBox(faceNet, frame):
    frameHeight = frame.shape[0] if len(frame.shape) == 3 else frame.shape[0]
    frameWidth = frame.shape[1] if len(frame.shape) == 3 else frame.shape[1]
    blob = cv2.dnn.blobFromImage(frame, 1.0, (227, 227), [104, 117, 123], swapRB=False)
    faceNet.setInput(blob)
    detection = faceNet.forward()
    faceBoxes = []
    for i in range(detection.shape[2]):
        confidence = detection[0, 0, i, 2]
        if confidence > 0.7:
            x1 = int(detection[0, 0, i, 3] * frameWidth)
            y1 = int(detection[0, 0, i, 4] * frameHeight)
            x2 = int(detection[0, 0, i, 5] * frameWidth)
            y2 = int(detection[0, 0, i, 6] * frameHeight)
            faceBoxes.append([x1, y1, x2, y2])
            cv2.rectangle(frame, (x1, y1), (x2, y2), (0, 255, 0), 1)
    return frame, faceBoxes


# Hàm xử lý sự kiện khi nhấn nút "Sử dụng Webcam"
def useWebcam():
    video_thread = Thread(target=processWebcam)
    video_thread.start()


# Hàm xử lý webcam trong một luồng riêng biệt
def processWebcam():
    video = cv2.VideoCapture(0)
    while True:
        hasFrame, vidFrame = video.read()
        if not hasFrame:
            break

        frame, faceBoxes = getFaceBox(faceNet, vidFrame)
        for faceBox in faceBoxes:
            face = frame[faceBox[1]:faceBox[3], faceBox[0]:faceBox[2]]
            blob = cv2.dnn.blobFromImage(face, 1.0, (227, 227), MODEL_MEAN_VALUES, swapRB=False)

            genderNet.setInput(blob)
            genderPred = genderNet.forward()
            gender = genderList[genderPred[0].argmax()]

            ageNet.setInput(blob)
            agePred = ageNet.forward()
            age = ageList[agePred[0].argmax()]

            labelGender = "{}".format("Gender : " + gender)
            labelAge = "{}".format("Age : " + age + "Years")
            cv2.putText(frame, labelGender, (faceBox[0], faceBox[1] - 40), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)
            cv2.putText(frame, labelAge, (faceBox[0], faceBox[1] - 10), cv2.FONT_HERSHEY_SIMPLEX, 0.8,
                        (0, 255, 255), 2, cv2.LINE_AA)

        showResultWebcam(frame)

        if cv2.waitKey(1) == ord('q'):
            break

    video.release()
    cv2.destroyAllWindows()


def processImage(image_path, root):
    print("File Path:", image_path)

    full_path = os.path.join(base_path, image_path)
    image = cv2.imread(full_path)

    if image is None:
        print(f"Error: Unable to read the image at '{full_path}'.")
        return

    if image.size == 0:
        print(f"Error: Image at '{full_path}' has size 0. Please check the file.")
        return

    frame, faceBoxes = getFaceBox(faceNet, image)

    for faceBox in faceBoxes:
        showResultImage(frame, image, root)


def showResultImage(frame, original_image, root):
    # Chuyển đổi ảnh gốc từ BGR sang RGB để sử dụng với Matplotlib
    original_image_rgb = cv2.cvtColor(original_image, cv2.COLOR_BGR2RGB)

    # Tạo subplot
    fig, axs = plt.subplots(1, 2, figsize=(10, 5))

    # Hiển thị ảnh gốc bên trái
    axs[0].imshow(original_image_rgb)
    axs[0].axis('off')
    axs[0].set_title('Original Image')

    # Hiển thị ảnh có nhận diện bên phải
    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    axs[1].imshow(frame_rgb)
    axs[1].axis('off')
    axs[1].set_title('Detected Image')

    # Hiển thị cả hai subplot
    plt.show()

def selectImage():
    file_path = filedialog.askopenfilename()
    if file_path:
        processImage(file_path, root)




def showResultWebcam(frame):
    cv2.imshow("Age-Gender Detector", frame)

root = tk.Tk()
root.title("Age-Gender Detection App")

# Tạo nút để sử dụng webcam
webcam_button = tk.Button(root, text="Sử dụng Webcam", command=useWebcam)
webcam_button.pack()

# Tạo nút để chọn ảnh
image_button = tk.Button(root, text="Chọn Ảnh", command=selectImage)
image_button.pack()

# Tạo nút để thoát chương trình
exit_button = tk.Button(root, text="Thoát", command=root.destroy)
exit_button.pack()

# Đường dẫn đến các mô hình và tệp tin cần thiết
faceProto = "opencv_face_detector.pbtxt"
faceModel = "opencv_face_detector_uint8.pb"
ageProto = "age_deploy.prototxt"
ageModel = "age_net.caffemodel"
genderProto = "gender_deploy.prototxt"
genderModel = "gender_net.caffemodel"

# Đọc các mô hình
faceNet = cv2.dnn.readNet(faceModel, faceProto)
ageNet = cv2.dnn.readNet(ageModel, ageProto)
genderNet = cv2.dnn.readNet(genderModel, genderProto)

# Danh sách giá trị trung bình của mô hình
MODEL_MEAN_VALUES = (78.4263377603, 87.7689143744, 114.895847746)
ageList = ['(0-2)', '(4-6)', '(8-12)', '(15-20)', '(25-32)', '(38-43)', '(48-53)', '(60-100)']
genderList = ['Male', 'Female']

# Tạo thuộc tính result_image_label và original_image_label cho đối tượng root
root.result_image_label = tk.Label(root)
root.result_image_label.pack()

root.original_image_label = tk.Label(root)
root.original_image_label.pack()

# Khởi chạy giao diện
root.mainloop()