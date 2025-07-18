from scipy import ndimage
import cv2
import numpy as np

############################################
# 3️⃣ Registration - Phase Correlation
############################################

def estimate_shift_phase_correlation(img1, img2):
    """
    Estimates (y, x) shift to align img2 to img1 using phase correlation.
    Returns shift as (y, x).
    """
    # OpenCV expects float32
    img1_f = np.float32(img1)
    img2_f = np.float32(img2)

    # Compute phase correlation
    shift, response = cv2.phaseCorrelate(img1_f, img2_f)
    print(f"[INFO] Estimated shift: x={shift[0]:.4f}, y={shift[1]:.4f}")
    return shift

############################################
# 4️⃣ Apply the estimated shift
############################################

def shift_image(img, shift):
    """
    Applies a sub-pixel shift to the image.
    Shift is (x, y).
    """
    shifted_img = ndimage.shift(img, shift[::-1], mode='reflect')
    return shifted_img

############################################
# Example usage / test
############################################

if __name__ == "__main__":
    # Load test images
    img1 = load_image_gray("LR1.png")
    img2 = load_image_gray("LR2.png")

    # Estimate shift
    shift = estimate_shift_phase_correlation(img1, img2)

    # Apply shift
    aligned_img2 = shift_image(img2, shift)

    # Visualize
    show_images([img1, img2, aligned_img2],
                ["LR Image 1 (Ref)", "LR Image 2 (Orig)", "LR Image 2 (Aligned)"])
