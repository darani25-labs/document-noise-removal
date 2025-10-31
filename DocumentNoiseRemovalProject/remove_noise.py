import cv2
import os

# --- Step 1: Load image in grayscale ---
img_path = 'image_file/noisy_document.jpg'  # Change path if needed
print(f"🔍 Reading image: {img_path}")

img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
if img is None:
    print("❌ Error: Could not load image. Check the path!")
    exit()
print("✅ Image loaded successfully")

# --- Step 2: Apply Non-local Means Denoising ---
# This method keeps edges and text sharp
denoised = cv2.fastNlMeansDenoising(img, None, h=15, templateWindowSize=7, searchWindowSize=21)
print("✅ Denoising completed")

# --- Step 3: (Optional) Light sharpening for text clarity ---
sharpen_kernel = cv2.getStructuringElement(cv2.MORPH_RECT, (1, 1))
sharpened = cv2.morphologyEx(denoised, cv2.MORPH_GRADIENT, sharpen_kernel)
result = cv2.addWeighted(denoised, 1.2, sharpened, -0.2, 0)
print("✅ Sharpening done")

# --- Step 4: Save output ---
os.makedirs('output', exist_ok=True)
output_path = 'output/noise_removed_clear.jpg'
cv2.imwrite(output_path, result)
print(f"✅ Clean document saved at: {output_path}")











