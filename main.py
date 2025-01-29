import cv2
import pytesseract
import numpy as np
from PIL import Image, ImageDraw, ImageFont

# No need to specify Tesseract path on macOS (Homebrew handles it)

def extract_text_from_balloons(image_path):
    """Extract text and bounding box locations from speech balloons in a comic book image."""
    image = cv2.imread(image_path)
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)

    # Apply thresholding for better contrast
    _, thresh = cv2.threshold(gray, 200, 255, cv2.THRESH_BINARY_INV)

    # Find contours (balloon detection)
    contours, _ = cv2.findContours(thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

    extracted_texts = []
    bounding_boxes = []

    for contour in contours:
        x, y, w, h = cv2.boundingRect(contour)

        # Extract region of interest (ROI)
        roi = gray[y:y+h, x:x+w]

        # Use Tesseract OCR to extract text
        text = pytesseract.image_to_string(roi, config="--psm 6").strip()

        if text:
            extracted_texts.append((x, y, w, h, text))
            bounding_boxes.append((x, y, w, h))

    return extracted_texts, bounding_boxes, image

def replace_text(image_path, new_texts):
    """Replace extracted text with new text while maintaining the speech balloon style."""
    extracted_texts, bounding_boxes, image = extract_text_from_balloons(image_path)
    image_pil = Image.open(image_path)
    draw = ImageDraw.Draw(image_pil)

    # Use a system font available on macOS
    try:
        font = ImageFont.truetype("/System/Library/Fonts/Supplemental/Arial.ttf", 20)
    except:
        font = ImageFont.load_default()

    for (x, y, w, h), new_text in zip(bounding_boxes, new_texts):
        # Draw a white rectangle to cover old text
        #draw.rectangle((x, y, x + w, y + h), fill="white")

        # Add new text
        draw.text((x + 5, y + 5), new_text, fill="black", font=font)

    image_pil.show()
    image_pil.save("updated_comic.png")

# Example usage
if __name__ =='__main__':

    image_path = "comic_sample.jpg"  # Replace with your image file
    new_texts = ["Hey Nihu!", "Yes tell me Dhyaan!"]  # Replace with your text
    replace_text(image_path, new_texts)
