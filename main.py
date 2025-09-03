import cv2
import cvzone
import mediapipe as mp
import numpy as np
import os
import math
import webbrowser
import subprocess
import time
import qrcode
from PIL import Image
from PIL import ImageFont, ImageDraw, Image
import win32print
import win32api
import requests
import threading
import pyautogui

API_URL = "http://localhost:3000/gesture"

screenshot_mode = False
qr_code_shown = False

def check_gesture():
    global screenshot_mode, qr_code_shown, printing_message_shown, printing_message_time

    while True:
        try:
            response = requests.get(API_URL)
            if response.status_code == 200:
                gesture_data = response.json()
                gesture = gesture_data.get("gesture", "")

                # Parsowanie gestu
                if "X001B[GEST=" in gesture:
                    gesture = gesture.split("X001B[GEST=")[-1].replace("]", "")

                # Obsługa gestów w trybie screenshot
                if screenshot_mode:
                    if qr_code_shown:
                        if gesture == "THUMB:DOWN":
                            print("Gest THUMB:DOWN - zamknięcie okna QR")
                            close_dialog()
                    else:
                        if gesture == "OPENPALM:UP":
                            print("Gest OPENPALM:UP - wybór kodu QR")
                            qr_code_shown = True
                            print("Kod QR zostanie wyświetlony")
                        elif gesture == "POINT:UP":
                            print("Gest POINT:UP - wybór druku")
                            try:
                                print("Drukowanie zdjęcia...")
                                printing_message_shown = True
                                printing_message_time = time.time()
                            except Exception as e:
                                print(f"Error printing: {e}")

                # Obsługa gestów w trybie normalnym
                else:
                    if gesture == "POINT:LEFT":
                        print("Wykonanie gestu POINT:LEFT - zmiana motywu w lewo")
                        change_theme_left()
                    elif gesture == "POINT:RIGHT":
                        print("Wykonanie gestu POINT:RIGHT - zmiana motywu w prawo")
                        change_theme_right()
                    elif gesture == "OK:UP":
                        print("Wykonanie gestu OK:UP - otwarcie okna dialogowego i screenshot")
                        open_dialog_and_screenshot()
                    elif gesture in ["NOGESTURE", "NOHAND"]:
                        print("Brak akcji dla gestu:", gesture)

            time.sleep(1)  # Sprawdzaj API co sekundę

        except Exception as e:
            print(f"Błąd podczas pobierania gestu: {e}")
            time.sleep(1000000) # ZMIENIC

# Funkcja zmiany motywu w lewo
def change_theme_left():
    global theme_index
    theme_index = (theme_index - 1) % len(themes)

# Funkcja zmiany motywu w prawo
def change_theme_right():
    global theme_index
    theme_index = (theme_index + 1) % len(themes)

# Funkcja otwierająca okno dialogowe i robiąca screenshot
def open_dialog_and_screenshot():
    global screenshot_mode, screenshot_clean_frame
    if save_screenshot(screenshot_clean_frame):
        screenshot_mode = True
        print("Screenshot wykonany! Otwieram okno dialogowe.")

# Funkcja zamykająca okno dialogowe
def close_dialog():
    global screenshot_mode, qr_code_shown, printing_message_shown
    screenshot_mode = False
    qr_code_shown = False
    printing_message_shown = False
    print("Zamknięto okno dialogowe.")

# Uruchomienie wątku sprawdzającego gesty
gesture_thread = threading.Thread(target=check_gesture, daemon=True)
gesture_thread.start()

RESOLUTION = (1280, 720)
# RESOLUTION = (3840, 2160)
smoothing_factor = 0.5

theme_backgrounds = {
    "burger.png": "background1.png",
    "dog.png": "background2.png",
    "creeper.png": "background3.png",
    "creeper2.png": "background3.png",
    "pig.png": "background3.png",
    "enderman.png": "background3.png",
    "steve.png": "background3.png",
    "pirate.png": "background4.png",
    "native.png": "background4.png",
    "sw.png": "background5.png"
}

themes = [ 
    "modification0.png", "burger.png", "dog.png", "beard.png", "cool.png", "creeper.png","creeper2.png","pig.png","enderman.png","steve.png",
    "star.png", "sunglass.png", "pirate.png", "native.png", "sw.png",
    "cat.png", "princess.png", "spiderman.png","batman.png","dragon.png",
    "modification1.png", "modification2.png", "modification3.png", "modification4.png"
]
thumbnails = {theme: cv2.resize(cv2.imread(theme, cv2.IMREAD_UNCHANGED), (90, 90)) for theme in themes}
overlays = {theme: cv2.imread(theme, cv2.IMREAD_UNCHANGED) for theme in themes if not theme.startswith("modification")}
backgrounds = {name: cv2.imread(f"backgrounds/{bg}", cv2.IMREAD_UNCHANGED) for name, bg in theme_backgrounds.items()}
title_image = cv2.imread("title.png", cv2.IMREAD_UNCHANGED)
left_arrow = cv2.imread("arrow_left.png", cv2.IMREAD_UNCHANGED)
right_arrow = cv2.imread("arrow_right.png", cv2.IMREAD_UNCHANGED)
photo_icon = cv2.imread("photo.png", cv2.IMREAD_UNCHANGED)
photo_icon_x = 1100  # <- dowolna wartość X
photo_icon_y = 570                   # <- dowolna wartość Y
photo_x_size = 150  # szerokość ikony aparatu
photo_y_size = 150  # wysokość ikony aparatu
photo_icon_resized = cv2.resize(photo_icon, (photo_x_size, photo_y_size))

cap = cv2.VideoCapture(0)
cap.set(3, RESOLUTION[0])
cap.set(4, RESOLUTION[1])
mp_face_mesh = mp.solutions.face_mesh
face_mesh = mp_face_mesh.FaceMesh(static_image_mode=False, max_num_faces=1)

screenshot_mode = False
qr_code_shown = False
printing_message_shown = False
printing_message_time = 0
screenshot_popup = None

from PIL import ImageFont, ImageDraw, Image

def draw_text_with_polish_chars(img, text, position, font_path="Speedee-Bold.ttf", font_size=40, color=(255,255,255)):
    # Konwertuj z OpenCV do Pillow (BGR → RGB)
    img_pil = Image.fromarray(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    draw = ImageDraw.Draw(img_pil)

    font = ImageFont.truetype(font_path, font_size)
    draw.text(position, text, font=font, fill=color)

    # Konwertuj z powrotem z Pillow do OpenCV (RGB → BGR)
    return cv2.cvtColor(np.array(img_pil), cv2.COLOR_RGB2BGR)

def lens_effect(frame):
    h, w = frame.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2, h / 2
    r = np.sqrt((map_x - cx) ** 2 + (map_y - cy) ** 2)
    max_r = np.sqrt(cx ** 2 + cy ** 2)
    scale = 3
    strength = 1 - (r / max_r)
    strength = np.clip(strength, 0, 1)
    map_x = ((map_x - cx) / (1 + strength * (scale - 1)) + cx).astype(np.float32)
    map_y = ((map_y - cy) / (1 + strength * (scale - 1)) + cy).astype(np.float32)
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR)


def spiral_effect(frame):
    h, w = frame.shape[:2]
    map_y, map_x = np.indices((h, w), dtype=np.float32)
    cx, cy = w / 2, h / 2
    dx, dy = map_x - cx, map_y - cy
    r = np.sqrt(dx**2 + dy**2)
    theta = np.arctan2(dy, dx)
    theta += 1.3 * (1 - r / r.max())
    map_x = r * np.cos(theta) + cx
    map_y = r * np.sin(theta) + cy
    return cv2.remap(frame, map_x, map_y, interpolation=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)

def stretch_x(frame):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (int(w * 2), h))
    return resized[:, (resized.shape[1] - w) // 2:(resized.shape[1] + w) // 2]

def stretch_y(frame):
    h, w = frame.shape[:2]
    resized = cv2.resize(frame, (w, int(h * 2)))
    return resized[(resized.shape[0] - h) // 2:(resized.shape[0] + h) // 2, :]

modification_modes = {
    "modification1.png": stretch_x,
    "modification2.png": stretch_y,
    "modification3.png": lens_effect,
    "modification4.png": spiral_effect
}

def rotate_image(image, angle):
    h, w = image.shape[:2]
    center = (w // 2, h // 2)

    # Oblicz nowy rozmiar tak, aby nic nie było ucięte
    cos = abs(np.cos(np.radians(angle)))
    sin = abs(np.sin(np.radians(angle)))
    new_w = int(h * sin + w * cos)
    new_h = int(h * cos + w * sin)

    # Oblicz macierz rotacji z uwzględnieniem przesunięcia środka
    M = cv2.getRotationMatrix2D(center, angle, 1.0)
    M[0, 2] += (new_w - w) / 2
    M[1, 2] += (new_h - h) / 2

    # Wykonaj rotację na powiększonym płótnie
    rotated = cv2.warpAffine(image, M, (new_w, new_h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_CONSTANT, borderValue=(0, 0, 0, 0))
    return rotated

def get_face_angle(landmarks, w, h):
    x1, y1 = int(landmarks[33].x * w), int(landmarks[33].y * h)
    x2, y2 = int(landmarks[263].x * w), int(landmarks[263].y * h)
    return -np.degrees(np.arctan2(y2 - y1, x2 - x1))

def save_screenshot(image):
    if not os.path.exists("public/photo"):
        os.makedirs("public/photo")
    
    # Usuń elementy GUI przed zapisem screenshotu
    clean_image = image.copy()
   
    
    cv2.imwrite("public/photo/screenshot.png", clean_image)
    print("Screenshot saved as public/photo/screenshot.png")
    return True

def generate_qr_code(url):
    qr = qrcode.QRCode(
        version=1,
        error_correction=qrcode.constants.ERROR_CORRECT_L,
        box_size=10,
        border=4,
    )
    qr.add_data(url)
    qr.make(fit=True)
    img = qr.make_image(fill_color="black", back_color="white")
    img = img.convert("RGB")
    return cv2.cvtColor(np.array(img), cv2.COLOR_RGB2BGR)

def mouse_callback(event, x, y, flags, param):
    global theme_index, screenshot_mode, qr_code_shown, printing_message_shown, printing_message_time
    
    if event == cv2.EVENT_LBUTTONDOWN:
        if screenshot_mode:
            if qr_button_x1 <= x <= qr_button_x2 and qr_button_y1 <= y <= qr_button_y2:
                qr_code_shown = True
            elif print_button_x1 <= x <= print_button_x2 and print_button_y1 <= y <= print_button_y2:
                try:
                    # Ścieżka do pliku
                    file_path = os.path.join(os.getcwd(), "public", "photo", "screenshot.png")

                    # Sprawdzenie, czy plik istnieje
                    if not os.path.exists(file_path):
                        print("Plik screenshot.png nie istnieje.")
                    else:
                        # Pobierz nazwę domyślnej drukarki
                        printer_name = win32print.GetDefaultPrinter()

                        # Wydrukuj plik za pomocą domyślnej aplikacji
                        win32api.ShellExecute(
                            0,
                            "print",
                            file_path,
                            None,
                            ".",
                            0
                        )
                        time.sleep(1)
                        #pyautogui.click(2650, 1785)
            
                        print(f"Drukowanie na drukarce: {printer_name}")

                        printing_message_shown = True
                        printing_message_time = time.time()
                except Exception as e:
                    print(f"Błąd podczas drukowania: {e}")
            elif close_button_x1 <= x <= close_button_x2 and close_button_y1 <= y <= close_button_y2:
                screenshot_mode = False
                qr_code_shown = False
                printing_message_shown = False
        else:
            if left_arrow_x <= x <= left_arrow_x + 150 and arrow_y_offset <= y <= arrow_y_offset + 200:
                theme_index = (theme_index - 1) % len(themes)
            elif right_arrow_x <= x <= right_arrow_x + 150 and arrow_y_offset <= y <= arrow_y_offset + 200:
                theme_index = (theme_index + 1) % len(themes)
            elif photo_icon_x <= x <= photo_icon_x + photo_x_size and photo_icon_y <= y <= photo_icon_y + photo_y_size:
                if save_screenshot(screenshot_clean_frame):
                    screenshot_mode = True
                    screenshot_popup = screenshot_clean_frame.copy()

cv2.namedWindow("EXON Faces", cv2.WND_PROP_FULLSCREEN)
cv2.setWindowProperty("EXON Faces", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
cv2.setMouseCallback("EXON Faces", mouse_callback)

theme_index = 0
smoothed_nose = smoothed_width = smoothed_height = smoothed_angle = None

while True:
    ret, frame_original = cap.read()
    if not ret:
        break

    screenshot_clean_frame = frame_original.copy()

    frame = frame_original.copy()
    theme = themes[theme_index]
    # 1. Nałóż tło (jeśli istnieje)
    bg = backgrounds.get(theme)
    if bg is not None:
        bg = cv2.resize(bg, (frame.shape[1], frame.shape[0]))
        frame = cvzone.overlayPNG(frame, bg, [0, 0])

    # 2. Zastosuj modyfikacje obrazu (jeśli istnieją)
    if theme in modification_modes:
        frame = modification_modes[theme](frame)
    else:
        # Przetwarzanie twarzy (jeśli nie ma modyfikacji)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = face_mesh.process(frame_rgb)
        if results.multi_face_landmarks:
            landmarks = results.multi_face_landmarks[0].landmark
            ih, iw = frame.shape[:2]
            nose_tip = (int(landmarks[1].x * iw), int(landmarks[1].y * ih))
            face_width = int(np.linalg.norm([(landmarks[127].x - landmarks[356].x) * iw, (landmarks[127].y - landmarks[356].y) * ih]))
            face_height = int(np.linalg.norm([(landmarks[152].x - landmarks[10].x) * iw, (landmarks[152].y - landmarks[10].y) * ih]))

            angle = get_face_angle(landmarks, iw, ih)
            if smoothed_nose is None:
                smoothed_nose, smoothed_width, smoothed_height, smoothed_angle = nose_tip, face_width, face_height, angle
            else:
                smoothed_nose = (int((1 - smoothing_factor) * smoothed_nose[0] + smoothing_factor * nose_tip[0]),
                                 int((1 - smoothing_factor) * smoothed_nose[1] + smoothing_factor * nose_tip[1]))
                smoothed_width = (1 - smoothing_factor) * smoothed_width + smoothing_factor * face_width
                smoothed_height = (1 - smoothing_factor) * smoothed_height + smoothing_factor * face_height
                smoothed_angle = (1 - smoothing_factor) * smoothed_angle + smoothing_factor * angle

            if theme in overlays:
                overlay = overlays[theme]
                theme_transformations = {
                    "burger.png": (1.6, 1.6, -20),
                    "dog.png": (1.8, 1.8, -20),
                    "beard.png": (2, 1.9, -50),
                    "cool.png": (1.8, 1.8, -40),
                    "creeper.png": (1.8, 1.8, -40),
                    "creeper2.png": (1.8, 1.8, -40),
                    "pig.png": (1.8, 1.8, -40),
                    "enderman.png": (1.8, 1.8, -40),
                    "steve.png": (1.8, 1.8, -40),
                    "sunglass.png": (1.5, 1.5, -40),
                    "pirate.png": (2, 1.7, -50),
                    "native.png": (2, 1.7, -50),
                    "princess.png": (2, 2, -80),
                    "spiderman.png": (1.8, 1.8, -40),
                    "batman.png": (2.2, 2.2, -40),
                    "sw.png": (2.2, 2.2, -40),
                    "dragon.png": (2.2, 2.2, -40),
                }

                picture_scale_X, picture_scale_Y, picture_offset_Y = theme_transformations.get(
                    theme, (1.8, 1.8, -20)
                )

                overlay_resized = cv2.resize(
                    overlay,
                    (
                        int(smoothed_width * picture_scale_X),
                        int(smoothed_height * picture_scale_Y),
                    ),
                )
                overlay_rotated = rotate_image(overlay_resized, smoothed_angle)
                x = smoothed_nose[0] - overlay_rotated.shape[1] // 2
                y = smoothed_nose[1] - overlay_rotated.shape[0] // 2 + picture_offset_Y
                frame = cvzone.overlayPNG(frame, overlay_rotated, [x, y])


    # 3. Nałóż elementy interfejsu (PO modyfikacjach)
    # Karuzela
    carousel_offset_x = 340
    carousel_offset_y = -10
    start_x = 50 + carousel_offset_x
    carousel_y = RESOLUTION[1] - 100 + carousel_offset_y
    for i in range(-2, 3):
        idx = (theme_index + i) % len(themes)
        thumb = thumbnails[themes[idx]]
        x = start_x + (i + 2) * (90 + 10)
        color = (0,199,247) if i == 0 else (10,70,6)
        cv2.rectangle(frame, (x - 5, carousel_y - 10), (x + 90 + 5, carousel_y + 90 + 10), color, -1)
        frame = cvzone.overlayPNG(frame, thumb, [x, carousel_y])

    # Strzałki
    arrow_y_offset = RESOLUTION[1] // 2 - 125 + 300 + 40
    left_arrow_x = 200
    right_arrow_x = 920
    frame = cvzone.overlayPNG(frame, left_arrow, [left_arrow_x, arrow_y_offset])
    frame = cvzone.overlayPNG(frame, right_arrow, [right_arrow_x, arrow_y_offset])

    # Napis title.png
    if title_image is not None:
        title_resized = cv2.resize(title_image, (800, 100))
        frame = cvzone.overlayPNG(frame, title_resized, [20, 20])

    # Przycisk screenshot
    screenshot_button_x1, screenshot_button_y1 = RESOLUTION[0] - 220, 20
    screenshot_button_x2, screenshot_button_y2 = RESOLUTION[0] - 20, 70
    photo_icon_resized = cv2.resize(photo_icon, (150, 150))  # lub inny rozmiar
    frame = cvzone.overlayPNG(frame, photo_icon_resized, [photo_icon_x, photo_icon_y])

    # 4. Obsługa trybu screenshot
    if screenshot_mode:
        popup = np.zeros((RESOLUTION[1], RESOLUTION[0], 3), dtype=np.uint8)
        popup[:] = (50, 50, 50)

        overlay = frame.copy()
        cv2.addWeighted(overlay, 0.3, popup, 0.7, 0, overlay)

        button_color = (194, 110, 31)  # kolor tła przycisków
        text_color = (255, 255, 255)   # kolor tekstu

        if qr_code_shown:
            qr_img = generate_qr_code("http://top-supple.co.uk/downloadphoto")
            qr_img = cv2.resize(qr_img, (400, 400))
            qr_x = (RESOLUTION[0] - qr_img.shape[1]) // 2
            qr_y = (RESOLUTION[1] - qr_img.shape[0]) // 2 - 50
            overlay[qr_y:qr_y + qr_img.shape[0], qr_x:qr_x + qr_img.shape[1]] = qr_img

            close_button_x1, close_button_y1 = (RESOLUTION[0] - 200) // 2, qr_y + qr_img.shape[0] + 30
            close_button_x2, close_button_y2 = (RESOLUTION[0] + 200) // 2, close_button_y1 + 50
            cv2.rectangle(overlay, (close_button_x1, close_button_y1), (close_button_x2, close_button_y2), button_color, -1)

            close_text = "Zamknij"
            text_size = cv2.getTextSize(close_text, cv2.FONT_HERSHEY_SIMPLEX, 1, 2)[0]
            text_x = (close_button_x1 + (200 - text_size[0]) // 2) - 10
            text_y = (close_button_y1 + (50 + text_size[1]) // 2) - 25
            overlay = draw_text_with_polish_chars(overlay, close_text, (text_x, text_y), font_path="Speedee-Bold.ttf", font_size=40, color=text_color)

        elif printing_message_shown:
            if time.time() - printing_message_time < 3:
                msg = "Drukowanie..."
                text_size = cv2.getTextSize(msg, cv2.FONT_HERSHEY_SIMPLEX, 1.5, 3)[0]
                text_x = ((RESOLUTION[0] - text_size[0]) // 2)
                text_y = ((RESOLUTION[1] + text_size[1]) // 2) - 60
                overlay = draw_text_with_polish_chars(overlay, msg, (text_x, text_y), font_path="Speedee-Bold.ttf", font_size=60, color=text_color)
            else:
                printing_message_shown = False
                screenshot_mode = False
        else:
            button_width = 500
            button_height = 100
            center_x = RESOLUTION[0] // 2

            qr_button_y1 = (RESOLUTION[1] // 2) - button_height - 15
            qr_button_y2 = qr_button_y1 + button_height
            print_button_y1 = qr_button_y2 + 30
            print_button_y2 = print_button_y1 + button_height

            qr_button_x1 = center_x - button_width // 2
            qr_button_x2 = center_x + button_width // 2
            print_button_x1 = qr_button_x1
            print_button_x2 = qr_button_x2

            cv2.rectangle(overlay, (qr_button_x1, qr_button_y1), (qr_button_x2, qr_button_y2), button_color, -1)
            qr_text = "Zapisz przez kod QR"
            text_size = cv2.getTextSize(qr_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = (qr_button_x1 + (button_width - text_size[0]) // 2) + 10
            text_y = (qr_button_y1 + (button_height + text_size[1]) // 2) - 30
            overlay = draw_text_with_polish_chars(overlay, qr_text, (text_x, text_y), font_path="Speedee-Bold.ttf", font_size=40, color=text_color)

            cv2.rectangle(overlay, (print_button_x1, print_button_y1), (print_button_x2, print_button_y2), button_color, -1)
            print_text = "Wydrukuj zdjęcie"
            text_size = cv2.getTextSize(print_text, cv2.FONT_HERSHEY_SIMPLEX, 1.2, 2)[0]
            text_x = (print_button_x1 + (button_width - text_size[0]) // 2) + 10
            text_y = (print_button_y1 + (button_height + text_size[1]) // 2) - 30
            overlay = draw_text_with_polish_chars(overlay, print_text, (text_x, text_y), font_path="Speedee-Bold.ttf", font_size=40, color=text_color)

        frame = overlay

    def fit_to_screen(frame, target_ratio=(16, 9)):
        target_w = RESOLUTION[0]
        target_h = RESOLUTION[1]
        current_h, current_w = frame.shape[:2]

        current_ratio = current_w / current_h
        desired_ratio = target_ratio[0] / target_ratio[1]

        if current_ratio > desired_ratio:
            new_w = int(current_h * desired_ratio)
            new_h = current_h
        else:
            new_w = current_w
            new_h = int(current_w / desired_ratio)

        resized_frame = cv2.resize(frame, (new_w, new_h))
        final_frame = np.zeros((target_h, target_w, 3), dtype=np.uint8)
        x_offset = (target_w - new_w) // 2
        y_offset = (target_h - new_h) // 2
        final_frame[y_offset:y_offset+new_h, x_offset:x_offset+new_w] = resized_frame
        return final_frame

    cv2.imshow("EXON Faces", fit_to_screen(frame))

    key = cv2.waitKey(1)
    if key == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()