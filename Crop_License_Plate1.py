from ultralytics import YOLO
import cv2
import numpy as np
import matplotlib.pyplot as plt
import os

image_path = "/home/ashkan/Desktop/Github/Iran-License-Plate-Reader/archive/train/images/338_png.rf.536d13ab8e710ce92b5d2288e7b3f7d4.jpg"

model = YOLO("runs/segment/train/weights/best.pt")

results = model(
    source=image_path,
    save=True,
    project="Main_Output",
    name="Predict_License_Plate"
)

img = cv2.imread(image_path)

image_name = os.path.splitext(os.path.basename(image_path))[0]

save_dir = "mask_crop"
os.makedirs(save_dir, exist_ok=True)

count = 0

for r in results:

    if r.masks is None:
        continue

    for poly in r.masks.xy:

        pts = np.array(poly, dtype=np.float32)

        mask = np.zeros(img.shape[:2], dtype=np.uint8)

        cv2.fillPoly(
            mask,
            [pts.astype(np.int32)],
            255
        )

        contours, _ = cv2.findContours(
            mask,
            cv2.RETR_EXTERNAL,
            cv2.CHAIN_APPROX_SIMPLE
        )

        if not contours:
            continue

        contour = max(
            contours,
            key=cv2.contourArea
        )

        epsilon = 0.02 * cv2.arcLength(
            contour,
            True
        )

        approx = cv2.approxPolyDP(
            contour,
            epsilon,
            True
        )

        if len(approx) != 4:

            rect = cv2.minAreaRect(
                contour
            )

            box = cv2.boxPoints(rect)

            approx = np.array(
                box,
                dtype=np.float32
            )

        else:

            approx = approx.reshape(
                4,
                2
            ).astype(np.float32)

        points = approx

        s = points.sum(axis=1)
        d = np.diff(points, axis=1).reshape(-1)

        top_left = points[np.argmin(s)]
        bottom_right = points[np.argmax(s)]
        top_right = points[np.argmin(d)]
        bottom_left = points[np.argmax(d)]

        ordered = np.array(
            [
                top_left,
                top_right,
                bottom_right,
                bottom_left
            ],
            dtype=np.float32
        )

        width_top = np.linalg.norm(
            ordered[1] - ordered[0]
        )

        width_bottom = np.linalg.norm(
            ordered[2] - ordered[3]
        )

        height_left = np.linalg.norm(
            ordered[3] - ordered[0]
        )

        height_right = np.linalg.norm(
            ordered[2] - ordered[1]
        )

        width = int(
            max(
                width_top,
                width_bottom
            )
        )

        height = int(
            max(
                height_left,
                height_right
            )
        )

        if width <= 0 or height <= 0:
            continue

        dst = np.array(
            [
                [0, 0],
                [width - 1, 0],
                [width - 1, height - 1],
                [0, height - 1]
            ],
            dtype=np.float32
        )

        M = cv2.getPerspectiveTransform(
            ordered,
            dst
        )

        plate = cv2.warpPerspective(
            img,
            M,
            (width, height),
            flags=cv2.INTER_CUBIC,
            borderMode=cv2.BORDER_REPLICATE
        )

        plate_mask = cv2.warpPerspective(
            mask,
            M,
            (width, height),
            flags=cv2.INTER_NEAREST
        )

        plate = cv2.bitwise_and(
            plate,
            plate,
            mask=plate_mask
        )

        gray = cv2.cvtColor(
            plate,
            cv2.COLOR_BGR2GRAY
        )

        gray = cv2.resize(
            gray,
            None,
            fx=3,
            fy=3,
            interpolation=cv2.INTER_CUBIC
        )

        gray = cv2.fastNlMeansDenoising(
            gray,
            None,
            8
        )

        clahe = cv2.createCLAHE(
            clipLimit=2.5,
            tileGridSize=(8, 8)
        )

        gray = clahe.apply(gray)

        blur = cv2.GaussianBlur(
            gray,
            (0, 0),
            1.2
        )

        gray = cv2.addWeighted(
            gray,
            1.5,
            blur,
            -0.5,
            0
        )

        save_path = os.path.join(
            save_dir,
            f"{image_name}_{count}.png"
        )

        cv2.imwrite(
            save_path,
            gray
        )

        plt.figure(figsize=(10, 3))
        plt.imshow(gray, cmap="gray")
        plt.axis("off")
        plt.show()

        count += 1