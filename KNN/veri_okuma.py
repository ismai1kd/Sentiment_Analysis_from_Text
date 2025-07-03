# Pandas kütüphanesini içeri aktarın
import pandas as pd

# CSV dosyasını okuyun
verikumesi = pd.read_csv("C:/Users/Ismail/Desktop/Emre_KNN/Emotion_final.csv")

# İlk 5 satırı yazdırın
print(verikumesi.head())

# Etiket sütununu (Emotion) seçin
etiketkumesi = verikumesi["Emotion"]
print(etiketkumesi)

# Özellik sütununu (Text) seçin
ozellikkumesi = verikumesi["Text"]
print(ozellikkumesi)

# train_test_split fonksiyonunu içeri aktarın
from sklearn.model_selection import train_test_split

# Veriyi eğitim ve test kümelerine ayırın
ozellikkumesi_train, ozellikkumesi_test, etiketkumesi_train, etiketkumesi_test = train_test_split(
    ozellikkumesi, etiketkumesi, test_size=0.01, random_state=0
)

# Metin verisini sayısal verilere dönüştürmek için TfidfVectorizer'ı kullanın
from sklearn.feature_extraction.text import TfidfVectorizer

# TfidfVectorizer'ı oluşturun ve eğitim verisine uygulayın
vectorizer = TfidfVectorizer()
ozellikkumesi_train_tfidf = vectorizer.fit_transform(ozellikkumesi_train)
ozellikkumesi_test_tfidf = vectorizer.transform(ozellikkumesi_test)

# KNN sınıflandırıcıyı içeri aktarın
from sklearn.neighbors import KNeighborsClassifier

# KNN modelini oluşturun ve eğitim verileriyle eğitin
knn = KNeighborsClassifier(n_neighbors=3, metric="euclidean")
knn.fit(ozellikkumesi_train_tfidf, etiketkumesi_train)

# Modelin tahmin sonuçlarını al
modelintahminsonucu = knn.predict(ozellikkumesi_test_tfidf)
print(modelintahminsonucu)

# Karışıklık Matrisi (Confusion Matrix) hesaplayın
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score

cm = confusion_matrix(etiketkumesi_test, modelintahminsonucu)
print("Karışıklık Matrisi:\n", cm)

# Doğruluk skoru (Accuracy) hesaplayın
dogruluk = accuracy_score(etiketkumesi_test, modelintahminsonucu)
print(f"Doğruluk (Accuracy): {dogruluk:.4f}")

# Precision, Recall, F1-Score ve daha fazlası için classification_report
print("Sınıflandırma Raporu (Precision, Recall, F1-Score):\n", classification_report(etiketkumesi_test, modelintahminsonucu))

# Sensitivity (Duyarlılık) ve Specificity hesapla
TP = cm[1, 1]  # Gerçek Pozitif
TN = cm[0, 0]  # Gerçek Negatif
FP = cm[0, 1]  # Yanlış Pozitif
FN = cm[1, 0]  # Yanlış Negatif

# Sensitivity (Recall) ve Specificity hesapla
sensitivity = TP / (TP + FN)  # Gerçek pozitif / (Gerçek pozitif + Yanlış negatif)
specificity = TN / (TN + FP)  # Gerçek negatif / (Gerçek negatif + Yanlış pozitif)

print(f"Sensitivity (Duyarlılık): {sensitivity:.4f}")
print(f"Specificity (Özgüllük): {specificity:.4f}")
