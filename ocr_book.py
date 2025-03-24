import os
import math
import re
import cv2
import numpy as np
from PIL import Image, ImageEnhance
import pytesseract
from pdf2image import convert_from_path
from spellchecker import SpellChecker

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

class ImagePreprocessor:
    def correct_skew(self, image):
        gray = image
        thresh = image

        edges = cv2.Canny(thresh, 50, 150, apertureSize=3)
        lines = cv2.HoughLinesP(edges, 1, np.pi / 180, 100, minLineLength=100, maxLineGap=10)

        if lines is None:
            return image

        angles = []
        for line in lines:
            for x1, y1, x2, y2 in line:
                angle = math.degrees(math.atan2(y2 - y1, x2 - x1))
                angles.append(angle)

        skew_angle = np.median(angles)

        (h, w) = image.shape[:2]
        center = (w // 2, h // 2)
        M = cv2.getRotationMatrix2D(center, skew_angle, 1.0)
        rotated = cv2.warpAffine(image, M, (w, h), flags=cv2.INTER_CUBIC, borderMode=cv2.BORDER_REPLICATE)
        return rotated


    def unsharp_mask(self, image, kernel_size=(5, 5), sigma=1.0, amount=1.0, threshold=0):
        """Return a sharpened version of the image, using an unsharp mask."""
        blurred = cv2.GaussianBlur(image, kernel_size, sigma)
        sharpened = float(amount + 1) * image - float(amount) * blurred
        sharpened = np.maximum(sharpened, np.zeros(sharpened.shape))
        sharpened = np.minimum(sharpened, 255 * np.ones(sharpened.shape))
        sharpened = sharpened.round().astype(np.uint8)
        if threshold > 0:
            low_contrast_mask = np.absolute(image - blurred) < threshold
            np.copyto(sharpened, image, where=low_contrast_mask)
        return sharpened

    def preprocess_image(self, image_path):
        picture_pil = Image.open(image_path)
        picture_pil = picture_pil.convert('RGB')
        picture = np.array(picture_pil)

        gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)

        # Odstranění šumu
        gray = cv2.medianBlur(gray, 3)

        # Binarizace (adaptivní prahování)
        thresh = cv2.adaptiveThreshold(gray, 255, cv2.ADAPTIVE_THRESH_GAUSSIAN_C,
                                      cv2.THRESH_BINARY, 11, 2)

        # Dilatace a eroze
        kernel = np.ones((2, 2), np.uint8)
        dilated = cv2.dilate(thresh, kernel, iterations=1)
        eroded = cv2.erode(dilated, kernel, iterations=1)

        # Unsharp masking
        sharpened = self.unsharp_mask(eroded)

        crop_margin = 20
        height, width = sharpened.shape
        cropped_thresh = sharpened[crop_margin:height-crop_margin, crop_margin:width-crop_margin]

        corrected_thresh = self.correct_skew(cropped_thresh)
        return corrected_thresh


class PageSegmenter:
    def find_book_contour(self, image):
        contours, _ = cv2.findContours(image, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
        book_contour = None
        max_area = 0

        for cnt in contours:
            area = cv2.contourArea(cnt)
            if area > 10000:
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True)
                vertices = len(approx)
                if 4 <= vertices <= 6:
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    if 0.5 <= aspect_ratio <= 2:
                        if area > max_area:
                            max_area = area
                            book_contour = approx
        return book_contour

    def crop_to_book_contour(self, image, book_contour):
        if book_contour is not None:
            x, y, w, h = cv2.boundingRect(book_contour)
            cropped_book_thresh = image[y:y+h, x:x+w]
            return cropped_book_thresh
        return image

    def enhance_image(self, image):
        enhancer_contrast = ImageEnhance.Contrast(image)
        image_enhanced_contrast = enhancer_contrast.enhance(1.5)
        enhancer_sharpness = ImageEnhance.Sharpness(image_enhanced_contrast)
        image_enhanced = enhancer_sharpness.enhance(2)
        return image_enhanced

    def split_page(self, image_pil):
        picture = image_pil
        # Normalizace před rozdělením
        img_np = np.array(picture.convert('L'))
        img_norm = cv2.normalize(img_np, None, 0, 255, cv2.NORM_MINMAX)
        picture = Image.fromarray(img_norm)

        width, height = picture.size
        half_width = width // 2
        reserve_left = 50
        reserve_right = 30

        left_page_pil = picture.crop((0, 0, half_width + reserve_left, height))
        right_page_pil = picture.crop((half_width + reserve_right, 0, width, height))
        return left_page_pil, right_page_pil


class TextExtractor:
    def __init__(self):
        self.spell = SpellChecker(language='cs')  # Inicializace SpellCheckeru

    def extract_text_from_image(self, image_pil, language='ces'):
        return pytesseract.image_to_string(image_pil, lang=language, config='--psm 1')

    def cleanup_ocr_text(self, text):
        lines = text.splitlines()
        cleaned_lines = []
        remove_next_line = False

        for i in range(len(lines)):
            line = lines[i].rstrip()

            if remove_next_line:
                remove_next_line = False
                continue

            if re.search(r'(?<!\S)strana\s+\d+(?!\S)', line, re.IGNORECASE):
                remove_next_line = True
                continue

            if line.endswith('-'):
                if i + 1 < len(lines):
                    next_line = lines[i + 1].lstrip()
                    if next_line and next_line[0].islower():
                        word_part1 = line[:-1]
                        word_part2 = next_line
                        combined_word = word_part1 + word_part2
                        cleaned_lines.append(combined_word)
                        continue

            line = re.sub(r'-\s*\n(\S)', r'-\1', line)
            line = re.sub(r'\s*\n+(?=[a-zěščřžýáíéúů])', ' ', line)

            # Odstranění řádků, které obsahují pouze interpunkci
            if re.match(r'^\W+$', line):
                continue

            cleaned_lines.append(line)

        # Spellchecking
        corrected_lines = []
        for line in cleaned_lines:
            words = line.split()
            corrected_words = []
            for word in words:
                word_clean = re.sub(r'[^\w\s]', '', word)  # Odstranění interpunkce

                if word_clean and word_clean.lower() not in self.spell:
                    corrected = self.spell.correction(word_clean.lower())
                    if corrected:
                        # Nahrazení původního slova opraveným, zachování velikosti písmen
                        if word_clean[0].isupper():
                            corrected = corrected.capitalize()
                        corrected_words.append(corrected)
                    else:
                        corrected_words.append(word)  # Ponechání původního
                else:
                    corrected_words.append(word)
            corrected_lines.append(" ".join(corrected_words))

        return "\n".join(corrected_lines)


    def join_split_sentences(self, text):
        sentences = re.split(r'(?<!\w\.\w.)(?<![A-Z][a-z]\.)(?<![A-Z]\.)(?<![0-9]\.)(?<=\.|\?|\!)\s', text)
        corrected_sentences = []
        previous_sentence = None

        for sentence in sentences:
            sentence = sentence.strip()
            if not sentence:
                continue

            if previous_sentence is not None and sentence[0].islower() and not previous_sentence.endswith(('.','?','!')):
              corrected_sentences[-1] += " " + sentence
            else:
                corrected_sentences.append(sentence)
                previous_sentence = sentence

        return " ".join(corrected_sentences)


class PdfToTextConverter:
    def __init__(self, pdf_path, output_dir='temp_images', first_page=1, last_page=None):
        self.pdf_path = pdf_path
        self.output_dir = output_dir
        self.first_page = first_page
        self.last_page = last_page
        os.makedirs(self.output_dir, exist_ok=True)
        self.image_preprocessor = ImagePreprocessor()
        self.page_segmenter = PageSegmenter()
        self.text_extractor = TextExtractor()
        self.total_text = ""
        self.page_counter = first_page

    def convert_pdf_to_images(self):
        if not self.first_page or not self.last_page:
            return convert_from_path(self.pdf_path)
        return convert_from_path(self.pdf_path, first_page=self.first_page, last_page=self.last_page)

    def process_page(self, page_image, page_index):
        print(f"Processing double-page {page_index + 1}...")
        image_path = os.path.join(self.output_dir, f'double_page_{page_index + 1}.png')
        page_image.save(image_path, 'PNG')

        try:
            preprocessed_doublepage_thresh = self.image_preprocessor.preprocess_image(image_path)

            book_contour = self.page_segmenter.find_book_contour(preprocessed_doublepage_thresh)
            cropped_doublepage_thresh = self.page_segmenter.crop_to_book_contour(preprocessed_doublepage_thresh, book_contour)
            cropped_doublepage_pil = Image.fromarray(cropped_doublepage_thresh)

            enhanced_doublepage_pil = self.page_segmenter.enhance_image(cropped_doublepage_pil)
            left_page_pil, right_page_pil = self.page_segmenter.split_page(enhanced_doublepage_pil)

            left_page_thresh = np.array(left_page_pil.convert('L'))
            right_page_thresh = np.array(right_page_pil.convert('L'))

            corrected_left_thresh = self.image_preprocessor.correct_skew(left_page_thresh)
            left_page_pil_rotated = Image.fromarray(corrected_left_thresh)

            corrected_right_thresh = self.image_preprocessor.correct_skew(right_page_thresh)
            right_page_pil_rotated = Image.fromarray(corrected_right_thresh)

            text_left = self.text_extractor.extract_text_from_image(left_page_pil_rotated)
            text_right = self.text_extractor.extract_text_from_image(right_page_pil_rotated)

            left_lines = text_left.splitlines(keepends=True)
            if left_lines:
                text_left = "".join(left_lines[1:])
            right_lines = text_right.splitlines(keepends=True)
            if right_lines:
                text_right = "".join(right_lines[1:])

            self.total_text += f"\n" + text_left + "\n\n"
            self.page_counter += 1
            self.total_text += f"\n" + text_right + "\n\n"
            self.page_counter += 1

        except pytesseract.TesseractNotFoundError:
            print("Error: Tesseract OCR is not installed or not in PATH.")
            raise
        except Exception as e:
            print(f"Error during processing double-page {page_index + 1}: {e}")
            raise

    def process_pdf_to_text(self):
        pages = self.convert_pdf_to_images()
        for i, page in enumerate(pages):
            self.process_page(page, i)

        cleaned_text = self.text_extractor.cleanup_ocr_text(self.total_text)
        final_text = self.text_extractor.join_split_sentences(cleaned_text)
        return final_text


if __name__ == '__main__':
    pdf_file_path = 'Zakazana_retorika.pdf'
    output_text_file = 'output_from_pdf.txt'

    converter = PdfToTextConverter(pdf_file_path, first_page=1, last_page=3)  # Zpracování prvních 3 stran
    try:
        extracted_text = converter.process_pdf_to_text()

        print("\n--- Total Recognized Text ---")
        print(extracted_text)

        with open(output_text_file, 'w', encoding='utf-8') as f:
            f.write(extracted_text)
        print(f"\nTotal text saved to '{output_text_file}'")

    except pytesseract.TesseractNotFoundError:
        print("Please ensure Tesseract OCR is installed and its path is correctly configured.")
    except Exception as e:
        print(f"An error occurred during PDF processing: {e}")