from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "archive/train/images/499_png.rf.1a4fe5b14fbe11301c9fa2d6801fc0af.jpg"

model = YOLO("runs/segment/train/weights/best.pt")

results = model(
    source=image_path,
    save=True,
    save_crop=True,
    project="Main_Output",
    name="Predict_License_Plate"
)

img = cv2.imread(image_path)

image_name = os.path.splitext(os.path.basename(image_path))[0]
save_dir = "mask_crop"

count = 0

for r in results:
    if r.masks is None:
        continue

    for poly in r.masks.xy:

        pts = np.array(poly, dtype=np.float32)

        rect = cv2.minAreaRect(pts)
        box = cv2.boxPoints(rect).astype(np.float32)

        width = int(rect[1][0])
        height = int(rect[1][1])

        if width < height:
            width, height = height, width

        dst = np.array([
            [0, 0],
            [width, 0],
            [width, height],
            [0, height]
        ], dtype=np.float32)

        s = box.sum(axis=1)
        diff = np.diff(box, axis=1)

        ordered = np.array([
            box[np.argmin(s)],
            box[np.argmin(diff)],
            box[np.argmax(s)],
            box[np.argmax(diff)]
        ], dtype=np.float32)

        M = cv2.getPerspectiveTransform(ordered, dst)
        plate = cv2.warpPerspective(img, M, (width, height))

        gray = cv2.cvtColor(plate, cv2.COLOR_BGR2GRAY)

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_LANCZOS4
        )

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            h=8
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8,8)
        )

        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(gray, (0,0), 1.2)

        gray = cv2.addWeighted(
            gray,
            1.5,
            blur,
            -0.5,
            0
        )

        plate = gray

        save_path = os.path.join(save_dir, f"{image_name}_{count}.png")
        cv2.imwrite(save_path, plate)

        plt.figure(figsize=(8, 3))
        plt.imshow(cv2.cvtColor(plate, cv2.COLOR_BGR2RGB))
        plt.axis("off")
        # plt.show()

        count += 1

