import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
from sklearn.feature_extraction.text import TfidfVectorizer
import matplotlib.pyplot as plt

# Veriyi yükle
veri_kumesi = pd.read_csv("Emotion_final.csv")
X = veri_kumesi['Text']
y = veri_kumesi['Emotion']

# Metni sayısal verilere dönüştür
vectorizer = TfidfVectorizer(stop_words='english')
X = vectorizer.fit_transform(X)

# Eğitim ve test setlerine ayıralım
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.3, random_state=42)

# Random Forest modelini oluştur
rf_model = RandomForestClassifier(n_estimators=100, random_state=42)
rf_model.fit(X_train, y_train)

# Modeli test et
y_pred = rf_model.predict(X_test)

# Sonuçları değerlendirelim
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluk oranı: {accuracy:.4f}")
print("Sınıflandırma Raporu:\n", classification_report(y_test, y_pred))

# Confusion matrix (karışıklık matrisi) ile sensitivity ve specificity hesapla
cm = confusion_matrix(y_test, y_pred)

# Sensitivity (Recall) ve Specificity hesaplamaları
sensitivity = cm[1, 1] / (cm[1, 1] + cm[1, 0])  # Gerçek pozitif / (Gerçek pozitif + Yanlış negatif)
specificity = cm[0, 0] / (cm[0, 0] + cm[0, 1])  # Gerçek negatif / (Gerçek negatif + Yanlış pozitif)

print(f"Sensitivity (Duyarlılık): {sensitivity:.4f}")
print(f"Specificity (Özgüllük): {specificity:.4f}")

# Özelliklerin önemini görselleştirelim
importances = rf_model.feature_importances_
indices = importances.argsort()[-10:][::-1]

plt.figure(figsize=(10, 6))
plt.barh(range(10), importances[indices], align='center')
plt.yticks(range(10), [vectorizer.get_feature_names_out()[i] for i in indices])
plt.xlabel('Özelliklerin Önemi')
plt.title('En Önemli 10 Özellik (Random Forest)')
plt.show()
