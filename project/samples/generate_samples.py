"""
Generate synthetic test images and ground truth masks for the Foreground Object Extraction Tool.
"""
import cv2
import numpy as np
import os

SAMPLES_DIR = os.path.dirname(os.path.abspath(__file__))
GROUND_TRUTH_DIR = os.path.join(SAMPLES_DIR, "ground_truth")

os.makedirs(GROUND_TRUTH_DIR, exist_ok=True)

def save_image(name, img, is_mask=False):
    """Save image to samples or ground_truth directory."""
    if is_mask:
        path = os.path.join(GROUND_TRUTH_DIR, f"{name}_mask.png")
    else:
        path = os.path.join(SAMPLES_DIR, f"{name}.png")
    cv2.imwrite(path, img)
    print(f"Saved: {path}")

# 1. Simple background: White square on solid black background
def create_simple():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[75:225, 75:225] = [255, 255, 255]  # White square in center
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[75:225, 75:225] = 255
    return img, mask

# 2. Cluttered background: Random noise background with a distinct object
def create_cluttered():
    np.random.seed(42)
    # Background: random noise
    img = np.random.randint(50, 200, (300, 300, 3), dtype=np.uint8)
    # Foreground: bright green circle
    center = (150, 150)
    radius = 80
    mask = np.zeros((300, 300), dtype=np.uint8)
    cv2.circle(mask, center, radius, 255, -1)
    # Draw circle on image
    cv2.circle(img, center, radius, (0, 255, 0), -1)
    return img, mask

# 3. Low contrast: Similar intensity foreground and background
def create_low_contrast():
    img = np.full((300, 300, 3), 128, dtype=np.uint8)  # Medium gray background
    # Slightly lighter square
    img[75:225, 75:225] = [160, 160, 160]
    mask = np.zeros((300, 300), dtype=np.uint8)
    mask[75:225, 75:225] = 255
    return img, mask

# 4. Person-like shape: Ellipse body + circle head
def create_person_like():
    img = np.zeros((400, 300, 3), dtype=np.uint8)
    mask = np.zeros((400, 300), dtype=np.uint8)
    
    # Body: ellipse
    cv2.ellipse(img, (150, 280), (70, 120), 0, 0, 360, (200, 150, 100), -1)
    cv2.ellipse(mask, (150, 280), (70, 120), 0, 0, 360, 255, -1)
    
    # Head: circle
    cv2.circle(img, (150, 120), 50, (200, 150, 100), -1)
    cv2.circle(mask, (150, 120), 50, 255, -1)
    
    # Arms
    cv2.ellipse(img, (70, 200), (20, 60), -30, 0, 360, (200, 150, 100), -1)
    cv2.ellipse(mask, (70, 200), (20, 60), -30, 0, 360, 255, -1)
    cv2.ellipse(img, (230, 200), (20, 60), 30, 0, 360, (200, 150, 100), -1)
    cv2.ellipse(mask, (230, 200), (20, 60), 30, 0, 360, 255, -1)
    
    return img, mask

# 5. Small object: Tiny circle in center
def create_small_object():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [50, 50, 50]  # Dark background
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Very small object (10px radius)
    cv2.circle(img, (150, 150), 10, (255, 255, 0), -1)  # Cyan dot
    cv2.circle(mask, (150, 150), 10, 255, -1)
    return img, mask

# 6. Transparent/translucent objects: Semi-transparent overlay simulation
def create_translucent():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [100, 100, 100]  # Gray background
    mask = np.zeros((300, 300), dtype=np.uint8)
    
    # Create a "glass-like" object - lighter with some transparency effect
    # We simulate by making the foreground color blend with background
    center = (150, 150)
    radius = 80
    
    # Create mask
    cv2.circle(mask, center, radius, 255, -1)
    
    # Foreground color (simulated glass - lighter than background)
    fg_color = np.array([180, 180, 200], dtype=np.uint8)
    bg_color = np.array([100, 100, 100], dtype=np.uint8)
    
    # Simulate alpha blending: 0.7 * fg + 0.3 * bg
    y, x = np.ogrid[:300, :300]
    dist = np.sqrt((x - center[0])**2 + (y - center[1])**2)
    in_circle = dist <= radius
    
    # Blend
    alpha = 0.7
    blended = (alpha * fg_color + (1 - alpha) * bg_color).astype(np.uint8)
    img[in_circle] = blended
    
    return img, mask

# 7. Multiple objects: Several distinct shapes
def create_multiple_objects():
    img = np.zeros((300, 400, 3), dtype=np.uint8)
    img[:] = [30, 30, 30]  # Dark background
    mask = np.zeros((300, 400), dtype=np.uint8)
    
    # Object 1: Red rectangle
    cv2.rectangle(img, (30, 30, 80, 80), (0, 0, 255), -1)
    cv2.rectangle(mask, (30, 30, 80, 80), 255, -1)
    
    # Object 2: Green circle
    cv2.circle(img, (200, 70), 40, (0, 255, 0), -1)
    cv2.circle(mask, (200, 70), 40, 255, -1)
    
    # Object 3: Blue triangle
    pts = np.array([[300, 30], [350, 110], [250, 110]], np.int32)
    cv2.fillPoly(img, [pts], (255, 0, 0))
    cv2.fillPoly(mask, [pts], 255)
    
    # Object 4: Yellow ellipse (bottom)
    cv2.ellipse(img, (150, 220), (60, 30), 0, 0, 360, (0, 255, 255), -1)
    cv2.ellipse(mask, (150, 220), (60, 30), 0, 0, 360, 255, -1)
    
    return img, mask

# 8. Edge case: Foreground touching border
def create_touching_border():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [50, 50, 50]
    mask = np.zeros((300, 300), dtype=np.uint8)
    
    # Large object touching top and left borders
    cv2.rectangle(img, (0, 0, 150, 150), (255, 200, 100), -1)
    cv2.rectangle(mask, (0, 0, 150, 150), 255, -1)
    
    # Another touching bottom-right
    cv2.rectangle(img, (180, 180, 120, 120), (100, 200, 255), -1)
    cv2.rectangle(mask, (180, 180, 120, 120), 255, -1)
    
    return img, mask

# 9. Edge case: Very thin object (line)
def create_thin_object():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    img[:] = [40, 40, 40]
    mask = np.zeros((300, 300), dtype=np.uint8)
    
    # Thin diagonal line (3px thick)
    cv2.line(img, (20, 280), (280, 20), (255, 255, 255), 3)
    cv2.line(mask, (20, 280), (280, 20), 255, 3)
    
    # Thin horizontal line
    cv2.line(img, (50, 150), (250, 150), (0, 255, 255), 2)
    cv2.line(mask, (50, 150), (250, 150), 255, 2)
    
    return img, mask

# 10. Gradient background with centered object
def create_gradient_bg():
    img = np.zeros((300, 300, 3), dtype=np.uint8)
    # Horizontal gradient
    for i in range(300):
        val = int(i * 255 / 300)
        img[:, i] = [val, val, val]
    
    mask = np.zeros((300, 300), dtype=np.uint8)
    # Circle in center - will be challenging for simple threshold
    cv2.circle(img, (150, 150), 70, (128, 128, 128), -1)  # Medium gray circle
    cv2.circle(mask, (150, 150), 70, 255, -1)
    
    return img, mask


def main():
    print("Generating synthetic test images...")
    
    generators = [
        ("01_simple", create_simple),
        ("02_cluttered", create_cluttered),
        ("03_low_contrast", create_low_contrast),
        ("04_person_like", create_person_like),
        ("05_small_object", create_small_object),
        ("06_translucent", create_translucent),
        ("07_multiple_objects", create_multiple_objects),
        ("08_touching_border", create_touching_border),
        ("09_thin_object", create_thin_object),
        ("10_gradient_bg", create_gradient_bg),
    ]
    
    for name, gen_func in generators:
        img, mask = gen_func()
        save_image(name, img, is_mask=False)
        save_image(name, mask, is_mask=True)
    
    print(f"\nGenerated {len(generators)} test image pairs.")
    print(f"Images: {SAMPLES_DIR}")
    print(f"Masks: {GROUND_TRUTH_DIR}")


if __name__ == "__main__":
    main()