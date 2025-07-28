#!/usr/bin/env python3
"""
Create test sample images for FaceFusion testing
Generates simple colored images with face-like patterns
"""

import numpy as np
from PIL import Image, ImageDraw

def create_test_image(color, filename):
    """Create a simple test image with a face-like pattern"""
    # Create image
    img = Image.new('RGB', (512, 512), color=color)
    draw = ImageDraw.Draw(img)
    
    # Draw face-like pattern
    # Face circle
    face_color = tuple(int(c * 0.9) for c in color)
    draw.ellipse([100, 100, 412, 412], fill=face_color)
    
    # Eyes
    eye_color = tuple(int(c * 0.3) for c in color)
    draw.ellipse([150, 200, 220, 270], fill=eye_color)
    draw.ellipse([292, 200, 362, 270], fill=eye_color)
    
    # Mouth
    draw.arc([180, 280, 332, 380], start=0, end=180, fill=eye_color, width=5)
    
    # Save image
    img.save(filename, quality=95)
    print(f"Created: {filename}")

# Create test images
create_test_image((255, 200, 180), 'input/source.jpg')  # Light skin tone
create_test_image((180, 200, 255), 'input/target.jpg')  # Blue tone

print("\nTest images created successfully!")
print("Note: These are simple test patterns. For better results, use real face photos.")