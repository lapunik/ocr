from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract
import os

pytesseract.pytesseract.tesseract_cmd = "C:\Program Files\Tesseract-OCR\\tesseract.exe" 

#cesta_k_pdf = 'LoR.pdf' 
cesta_k_pdf = 'Zakazana_retorika.pdf' 

temp_dir = 'temp_obrazky' 
os.makedirs(temp_dir, exist_ok=True)

first_page = 1
last_page = 2

pages = convert_from_path(cesta_k_pdf, first_page=first_page, last_page=last_page) 

celkovy_text = ""

for i, page in enumerate(pages):
        
    print(f"Zpracovávám stránku {i + 1}...")

    cesta_k_obrazku = os.path.join(temp_dir, f'strana_{i + 1}.png')
    page.save(cesta_k_obrazku, 'PNG')

    try:
        picture = Image.open(cesta_k_obrazku)
        enhancer = ImageEnhance.Contrast(picture)
        picture = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(picture)
        picture = enhancer.enhance(2)

        text = pytesseract.image_to_string(picture, lang='ces')
        celkovy_text += text + "\n\n"

    except pytesseract.TesseractNotFoundError:
        print("Chyba: Tesseract OCR nebyl nalezen. Ujistěte se, že je nainstalován.")
        break  

    except Exception as e:
        print(f"Chyba při OCR na straně {i + 1}: {e}")

print("\n--- Celkový rozpoznaný text ---")
print(celkovy_text)

vystupni_soubor = 'vystup_z_pdf.txt'
with open(vystupni_soubor, 'w', encoding='utf-8') as f:
    f.write(celkovy_text)
print(f"\nCelkový text byl uložen do souboru '{vystupni_soubor}'")