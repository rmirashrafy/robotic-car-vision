import cv2
import numpy as np

# --- Load image ---
image_path = "C:/Users/Fatemeh/Music/car/car_simu-main/car_simu-main/i.png"
image = cv2.imread(image_path)
if image is None:
    print("Error: Image not found.")
    exit()

# --- Convert to grayscale ---
gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

# --- Detect high contrast regions using Laplacian ---
laplacian = cv2.Laplacian(gray, cv2.CV_64F)
laplacian_abs = np.uint8(np.absolute(laplacian))

# --- Threshold to detect sharp contrast (glare/high reflectivity) ---
_, high_contrast_mask = cv2.threshold(laplacian_abs, 40, 255, cv2.THRESH_BINARY)

# --- Optional: clean mask (remove noise) ---
kernel = cv2.getStructuringElement(cv2.MORPH_ELLIPSE, (5, 5))
high_contrast_mask = cv2.dilate(high_contrast_mask, kernel, iterations=1)
high_contrast_mask = cv2.erode(high_contrast_mask, kernel, iterations=1)

# --- Step 1: Make glare/contrast regions black in the original image ---
image_masked = image.copy()
image_masked[high_contrast_mask == 255] = [0, 0, 0]  # Set white mask pixels to black

# --- Step 2: Inpaint the black areas using original mask ---
inpainted = cv2.inpaint(image_masked, high_contrast_mask, inpaintRadius=3, flags=cv2.INPAINT_TELEA)

# --- Save outputs ---
cv2.imwrite("C:/Users/Fatemeh/Music/car/car_simu-main/car_simu-main/high_contrast_mask.png", high_contrast_mask)
cv2.imwrite("C:/Users/Fatemeh/Music/car/car_simu-main/car_simu-main/blacked_out_image.png", image_masked)
cv2.imwrite("C:/Users/Fatemeh/Music/car/car_simu-main/car_simu-main/inpainted_image.png", inpainted)

print("✅ High contrast areas blacked out and inpainted. All images saved.")
