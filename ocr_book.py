from pdf2image import convert_from_path
from PIL import Image, ImageEnhance
import pytesseract
import os
import cv2 # Import OpenCV
import numpy as np # Import NumPy

pytesseract.pytesseract.tesseract_cmd = "C:\\Program Files\\Tesseract-OCR\\tesseract.exe"

#cesta_k_pdf = 'LoR.pdf'
cesta_k_pdf = 'Zakazana_retorika.pdf'

temp_dir = 'temp_obrazky'
os.makedirs(temp_dir, exist_ok=True)

first_page = 1
last_page = 5

pages = convert_from_path(cesta_k_pdf, first_page=first_page, last_page=last_page)

celkovy_text = ""
strana_citac = first_page

for i, page in enumerate(pages):

    print(f"Zpracovávám dvojstranu {i + 1}...")

    cesta_k_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}.png')
    page.save(cesta_k_obrazku, 'PNG')
    # print(f"  Ukládám původní obrázek dvojstrany do '{cesta_k_obrazku}'...")

    try:
        picture_pil = Image.open(cesta_k_obrazku)
        picture_pil = picture_pil.convert('RGB') # Převod na RGB, pokud není
        picture = np.array(picture_pil) # Převod PIL Image na NumPy array pro OpenCV

        # 1. Převod na stupně šedi
        gray = cv2.cvtColor(picture, cv2.COLOR_RGB2GRAY)
        # cesta_k_sedemu_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_seda.png')
        # Image.fromarray(gray).save(cesta_k_sedemu_obrazku, 'PNG') # Uložení šedého obrázku
        # print(f"  Ukládám šedý obrázek do '{cesta_k_sedemu_obrazku}'...")


        # 2. Prahování (Otsuova metoda pro automatický práh)
        thresh = cv2.threshold(gray, 0, 255, cv2.THRESH_BINARY + cv2.THRESH_OTSU)[1]
        # cesta_k_prahovanemu_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany.png')
        # Image.fromarray(thresh).save(cesta_k_prahovanemu_obrazku, 'PNG') # Uložení prahovaného obrázku
        # print(f"  Ukládám prahovaný obrázek do '{cesta_k_prahovanemu_obrazku}'...")

        # 2.1. Fixní ořez prahového obrázku (20 pixelů z každé strany)
        crop_margin = 20
        height, width = thresh.shape
        cropped_thresh = thresh[crop_margin:height-crop_margin, crop_margin:width-crop_margin]
        # cesta_k_orezaneho_prahovaneho_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany_orezany.png')
        # Image.fromarray(cropped_thresh).save(cesta_k_orezaneho_prahovaneho_obrazku, 'PNG') # Uložení ořezaného prahovaného obrázku
        # print(f"  Ukládám ořezaný prahovaný obrázek do '{cesta_k_orezaneho_prahovaneho_obrazku}'...")

        # *** Nyní pracujeme jen s OŘEZANÝM PRAHOVANÝM OBRÁZKEM (cropped_thresh) ***

        # 3. Hledání obrysů (vnějších) - na OŘEZANÉM PRAHOVANÉM OBRÁZKU
        contours, hierarchy = cv2.findContours(cropped_thresh, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)

        # *** DEBUG: Vykreslení obrysů přímo na OŘEZANÝ PRAHOVANÝ obrázek ***
        thresh_with_contours = cropped_thresh.copy() # Kopie pro vykreslení obrysů
        thresh_with_contours = cv2.cvtColor(thresh_with_contours, cv2.COLOR_GRAY2RGB) # Pro barevné obrysy
        book_contour = None
        other_contours_to_draw = [] # List pro obrysy, které NEJSOU knihou

        # 4. Filtrování obrysů - hledání největšího obdélníkového obrysu (předpoklad knihy)
        max_area = 0
        for cnt in contours:
            area = cv2.contourArea(cnt)
            # print(f"    Obrys - Plocha: {area}") # DEBUG: Výpis plochy obrysu
            if area > 10000: # Přidáno minimální plocha, aby se zamezilo malým šumovým obrysům, uprav dle potřeby
                perimeter = cv2.arcLength(cnt, True)
                approx = cv2.approxPolyDP(cnt, 0.04 * perimeter, True) # Aproximace obrysu polygonem
                vertices = len(approx)
                # print(f"      Aproximace - Počet vrcholů: {vertices}") # DEBUG: Výpis počtu vrcholů
                if 4 <= vertices <= 6: # Obdélník má 4 až šest vrcholů (4 = obdélník, 5 a 6 = obdélník s zaoblenými rohy)
                    x, y, w, h = cv2.boundingRect(approx)
                    aspect_ratio = float(w) / h
                    # print(f"        Obdélník - Poměr stran: {aspect_ratio:.2f}") # DEBUG: Výpis poměru stran
                    if 0.5 <= aspect_ratio <= 2: # Rozumný poměr stran pro knihu (orientační)
                        max_area = area
                        book_contour = approx
                        # print(f"          *** Vybrán jako obrys knihy ***") # DEBUG: Potvrzení výběru
                    else:
                        # print(f"          --- Poměr stran mimo rozsah (0.5-2), obrys ZAMÍTNUT.") # DEBUG: Důvod zamítnutí
                        other_contours_to_draw.append(cnt) # Uložíme pro vykreslení zeleně
                else:
                    # print(f"          --- Počet vrcholů != 4, obrys ZAMÍTNUT.") # DEBUG: Důvod zamítnutí
                    other_contours_to_draw.append(cnt) # Uložíme pro vykreslení zeleně
            else:
                # print(f"      --- Plocha < 10000, obrys ZAMÍTNUT.") # DEBUG: Důvod zamítnutí
                pass

        # *** DEBUG: Vykreslení obrysů - ČERVENĚ = obrys knihy, ZELENĚ = ostatní filtrované obrysy ***
        if book_contour is not None:
            cv2.drawContours(thresh_with_contours, [book_contour], -1, (0, 0, 255), 2) # Kreslíme ČERVENĚ obrys knihy
        cv2.drawContours(thresh_with_contours, other_contours_to_draw, -1, (0, 255, 0), 2) # Kreslíme ZELENĚ ostatní obrysy
        # cesta_k_obrysům_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany_orezany_obrysy.png')
        # Image.fromarray(thresh_with_contours).save(cesta_k_obrysům_obrazku, 'PNG')
        # print(f"  Ukládám OŘEZANÝ PRAHOVANÝ obrázek s vyznačenými obrysy (ČERVENĚ=kniha, ZELENĚ=ostatní) do '{cesta_k_obrysům_obrazku}'...")


        cropped_picture_pil = None # Inicializace pro případ, že není detekován book_contour
        if book_contour is not None:
            # 5. Ořezání - Nyní ořezáváme OŘEZANÝ PRAHOVANÝ obrázek
            x, y, w, h = cv2.boundingRect(book_contour)
            # print(f"  Ořezová oblast - x:{x}, y:{y}, w:{w}, h:{h}") # DEBUG: Výpis ořezové oblasti
            cropped_book_thresh = cropped_thresh[y:y+h, x:x+w] # Ořezáváme cropped_thresh
            cropped_picture_pil = Image.fromarray(cropped_book_thresh) # Převedeme na PIL Image pro další zpracování

            # cesta_k_dalsimu_orezaneho_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany_orezany_kniha.png')
            # cropped_picture_pil.save(cesta_k_dalsimu_orezaneho_obrazku, 'PNG') # Uložení ořezaného obrázku knihy
            # print(f"  Detekována plocha knihy a OŘEZANÝ PRAHOVANÝ obrázek ořezán PODRUHÉ. Ukládám ořezaný obrázek do '{cesta_k_dalsimu_orezaneho_obrazku}'...")


        else:
            # print("  Plocha knihy nebyla detekována po fixním ořezu. Používám POUZE fixně ořezaný prahovaný obrázek.")
            cropped_picture_pil = Image.fromarray(cropped_thresh) # Použijeme jen fixně ořezaný prahový obrázek
            # cesta_k_dalsimu_orezaneho_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany_orezany_pouzity.png')
            # cropped_picture_pil.save(cesta_k_dalsimu_orezaneho_obrazku, 'PNG')
            # print(f"  Ukládám fixně ořezaný prahovaný obrázek jako použitý do '{cesta_k_dalsimu_orezaneho_obrazku}'...")


        # 6. Vylepšení obrázku - pracujeme s ořezaným PRAHOVANÝM obrázkem (cropped_picture_pil)
        picture_pil = cropped_picture_pil # Pro zjednodušení názvu proměnné, nyní picture_pil je ořezaný prahovaný obrázek
        enhancer = ImageEnhance.Contrast(picture_pil)
        picture_pil = enhancer.enhance(1.5)
        enhancer = ImageEnhance.Sharpness(picture_pil)
        picture_pil = enhancer.enhance(2)
        # cesta_k_vylepseneho_obrazku = os.path.join(temp_dir, f'dvojstrana_{i + 1}_prahovany_orezany_vylepseny.png')
        # picture_pil.save(cesta_k_vylepseneho_obrazku, 'PNG') # Uložení vylepšeného obrázku
        # print(f"  Ukládám vylepšený OŘEZANÝ PRAHOVANÝ obrázek do '{cesta_k_vylepseneho_obrazku}'...")


        # 7. Rozdělení obrázku na dvě poloviny - pracujeme s vylepšeným OŘEZANÝM PRAHOVANÝM obrázkem
        picture = picture_pil # Pro zjednodušení názvu, picture je nyní vylepšený ořezaný prahovaný obrázek
        width, height = picture.size
        half_width = width // 2
        reserve_left = 50 # Rezerva pro případ, že by byl pruh mezi stránkami (např. u vazby)
        reserve_right = 30 # 

        # Levá strana (první stránka)
        left_page = picture.crop((0, 0, half_width + reserve_left, height))
        # *** Přidáno ukládání levé strany ***
        # cesta_k_leve_strane = os.path.join(temp_dir, f'strana_{strana_citac}_leva.png')
        # left_page.save(cesta_k_leve_strane, 'PNG')
        # print(f"  Rozpoznávám text pro stránku {strana_citac}") # (levá strana), ukládám obrázek do '{cesta_k_leve_strane}'...")
        text_left = pytesseract.image_to_string(left_page, lang='ces')
        celkovy_text += f"Strana {strana_citac}:\n" + text_left + "\n\n"
        strana_citac += 1

        # Pravá strana (druhá stránka)
        right_page = picture.crop((half_width + reserve_right, 0, width, height))
        # *** Přidáno ukládání pravé strany ***
        # cesta_k_prave_strane = os.path.join(temp_dir, f'strana_{strana_citac}_prava.png')
        # right_page.save(cesta_k_prave_strane, 'PNG')
        # print(f"  Rozpoznávám text pro stránku {strana_citac}") # (pravá strana), ukládám obrázek do '{cesta_k_prave_strane}'...")
        text_right = pytesseract.image_to_string(right_page, lang='ces')
        celkovy_text += f"Strana {strana_citac}:\n" + text_right + "\n\n"
        strana_citac += 1

    except pytesseract.TesseractNotFoundError:
        print("Chyba: Tesseract OCR nebyl nalezen. Ujistěte se, že je nainstalován.")
        break

    except Exception as e:
        print(f"Chyba při OCR na dvojstraně {i + 1}: {e}")

print("\n--- Celkový rozpoznaný text ---")
print(celkovy_text)

vystupni_soubor = 'vystup_z_pdf.txt'
with open(vystupni_soubor, 'w', encoding='utf-8') as f:
    f.write(celkovy_text)
print(f"\nCelkový text byl uložen do souboru '{vystupni_soubor}'")