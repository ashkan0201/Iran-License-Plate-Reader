from ultralytics import YOLO
import cv2
import numpy as np
import os
import matplotlib.pyplot as plt

model = YOLO("runs/segment/train/weights/best.pt")

img_path = "C:/Users/user/OneDrive/Desktop/sample-plate-dataset-5000/images/13.jpg"

results = model.predict(
    img_path,
    conf=0.25,
    verbose=False,
    save=True
)

img = cv2.imread(img_path)

os.makedirs("mask_crop", exist_ok=True)

base_name = os.path.splitext(os.path.basename(img_path))[0]

for r in results:
    if r.masks is None:
        continue

    for i, mask in enumerate(r.masks.data.cpu().numpy()):

        mask = cv2.resize(
            mask.astype(np.uint8),
            (img.shape[1], img.shape[0]),
            interpolation=cv2.INTER_NEAREST
        )

        result = cv2.bitwise_and(img, img, mask=mask)
        ys, xs = np.where(mask > 0)

        if len(xs) == 0:
            continue

        x1, x2 = xs.min(), xs.max()
        y1, y2 = ys.min(), ys.max()

        plate_only = result[y1:y2, x1:x2]

        file_name = os.path.basename(img_path)

        cv2.imwrite(
            f"mask_crop/{file_name}_plate_{i}.png",
            plate_only
        )

print("done")


# img_output_no_crp = plt.imread("")
# img_output_with_crp = plt.imread("")

# plt.figure(figsize=(10,5))

# plt.subplot(1,2,1)
# plt.imshow(img_output_no_crp)
# plt.title("YOLO Output")
# plt.axis("off")

# plt.subplot(1,2,2)
# plt.imshow(img_output_with_crp)
# plt.title("Only Plate")
# plt.axis("off")

# plt.show()