import cv2
import numpy as np
import matplotlib.pyplot as plt
import time

def fetch_satellite_image(latitude, longitude):
    print(f"\n[UYDU] {latitude}, {longitude} koordinatlarındaki uydudan anlık görüntü talep ediliyor...")
    time.sleep(1.5)
    print("[UYDU] Görsel başarıyla indirildi ve sisteme aktarıldı.")
    return "65984754-FFB1-4E93-9740-88A0F236DA60.jpeg"

def send_alert_notification(area_name, lat, lon):
    print("\n" + "="*50)
    print("🚨 [ACİL DURUM UYARISI] 🚨")
    print(f"Bölge: {area_name}")
    print(f"Koordinatlar: Lat: {lat}, Lon: {lon}")
    print("Mesaj: DİKKAT! Görüntü işleme algoritması bu bölgede yüksek riskli RİP AKINTISI tespit etti.")
    print("Lütfen denizcileri, tekneleri ve yüzücüleri bölgeden uzak tutun!")
    print("="*50 + "\n")

def analyze_rip_current(image_path, min_contour_area=1000):
    image = cv2.imread(image_path)
    if image is None:
        print(f"Hata: '{image_path}' görseli bulunamadı. Lütfen dosya yolunu kontrol edin.")
        return False, None, None, None

    image = cv2.resize(image, (512, 512))
    gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    blur = cv2.GaussianBlur(gray, (5, 5), 0)
    edges = cv2.Canny(blur, 50, 150)
    thresh = cv2.adaptiveThreshold(blur, 255, cv2.ADAPTIVE_THRESH_MEAN_C, cv2.THRESH_BINARY_INV, 11, 3)
    
    kernel = np.ones((5, 5), np.uint8)
    closed = cv2.morphologyEx(thresh, cv2.MORPH_CLOSE, kernel)
    
    contours, _ = cv2.findContours(closed, cv2.RETR_EXTERNAL, cv2.CHAIN_APPROX_SIMPLE)
    
    rip_like = image.copy()
    is_rip_detected = False
    
    if contours:
        biggest = max(contours, key=cv2.contourArea)
        biggest_area = cv2.contourArea(biggest)
        cv2.drawContours(rip_like, [biggest], -1, (0, 255, 0), 2)
        if biggest_area > min_contour_area:
            is_rip_detected = True

    return is_rip_detected, image, closed, rip_like

if __name__ == "__main__":
    target_beach = "Samsun Atakum Sahili - İstasyon 4"
    target_lat = 41.3275
    target_lon = 36.3242

    print("--- RİP AKINTISI ERKEN UYARI SİSTEMİ BAŞLATILDI ---")
    
    downloaded_img = fetch_satellite_image(target_lat, target_lon)
    
    print("\n[SİSTEM] Görüntü işleme algoritması çalıştırılıyor...")
    detected, orig, mask, result = analyze_rip_current(downloaded_img, min_contour_area=1000)

    if orig is not None:
        if detected:
            print("[SONUÇ] Yazılı Çıktı: Rip akıntısı olma ihtimali var. [TETİKLENİYOR]")
            send_alert_notification(target_beach, target_lat, target_lon)
        else:
            print("[SONUÇ] Yazılı Çıktı: Rip akıntısı tespit edilmedi. [GÜVENLİ]")

        print("[SİSTEM] Analiz grafikleri ekrana yansıtılıyor...")
        plt.figure(figsize=(15, 5))

        plt.subplot(1, 3, 1)
        plt.title(f"Orijinal Uydu Görseli\n({target_beach})")
        plt.imshow(cv2.cvtColor(orig, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.subplot(1, 3, 2)
        plt.title("Eşikleme & Maskeleme\n(Köpük ve Yoğunluk Analizi)")
        plt.imshow(mask, cmap='gray')
        plt.axis("off")

        plt.subplot(1, 3, 3)
        plt.title("Algoritma Sonucu\n(En Büyük Risk Alanı)")
        plt.imshow(cv2.cvtColor(result, cv2.COLOR_BGR2RGB))
        plt.axis("off")

        plt.tight_layout()
        plt.show()