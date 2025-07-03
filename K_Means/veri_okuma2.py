import pandas as pd
import matplotlib.pyplot as plt
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    classification_report,
    silhouette_score,
)
from sklearn.decomposition import PCA
import numpy as np
import warnings

# Uyarıları susturmak için
warnings.filterwarnings("ignore", category=UserWarning)
warnings.filterwarnings("ignore", category=FutureWarning)

# Veri kümesini okuyalım
veri_kumesi = pd.read_csv("Emotion_final.csv")

# Sadece 'Text' sütununu alalım
metinler = veri_kumesi["Text"]

# Doğru etiketleri (Ground truth)
etiketler = veri_kumesi["Emotion"]

# Etiketleri sayısal değere dönüştürelim
etiketler_kodlu, unique_labels = pd.factorize(etiketler)

# TF-IDF vektörleştirme
vectorizer = TfidfVectorizer(
    stop_words="english",
    ngram_range=(1, 3),  # Trigramları da dahil ettik
    max_df=0.95,         # Daha yüksek frekanslı kelimeleri dahil ettik
    min_df=3,            # Daha düşük frekanslı kelimeleri dahil ettik
)
v = vectorizer.fit_transform(metinler)

# Elbow yöntemi ve silhouette skor ile en iyi küme sayısını belirleme
distortions = []
sil_scores = []
for k in range(2, 25):  # Daha geniş bir aralıkta k sayısını deneyelim
    km = KMeans(n_clusters=k, init="k-means++", max_iter=500, random_state=0)  # max_iter artırıldı
    km.fit(v)
    distortions.append(km.inertia_)
    sil_scores.append(silhouette_score(v, km.labels_))

# Elbow yöntemi ve silhouette skor görselleştirme
plt.figure(figsize=(10, 5))
plt.plot(range(2, 25), distortions, marker="o", label="Distortion")
plt.plot(range(2, 25), sil_scores, marker="x", label="Silhouette Score")
plt.xlabel("Number of Clusters")
plt.ylabel("Score")
plt.legend()
plt.title("Elbow Method & Silhouette Score")
plt.show()

# K-Means modelleme (optimal küme sayısıyla)
optimal_clusters = 8  # Yeni analizde seçilmiş en iyi küme sayısı
km = KMeans(n_clusters=optimal_clusters, init="k-means++", max_iter=500, random_state=0)
km.fit(v)

# Küme tahminleri
kume_tahminleri = km.predict(v)

# Küme etiketlerini doğru etiketlere eşleme
kume_etiket_map = {}
for i in range(optimal_clusters):  # Tahmin edilen tüm kümeleri kontrol et
    mask = (kume_tahminleri == i)
    if np.sum(mask) > 0:
        kume_etiket_map[i] = np.bincount(etiketler_kodlu[mask]).argmax()

# Tahmini küme etiketlerini doğru etikete çevir
# Sözlükte olmayan küme etiketleri için varsayılan bir değer ekle (örneğin -1)
kume_tahminleri_mapped = [kume_etiket_map.get(label, -1) for label in kume_tahminleri]


# Tahmini küme etiketlerini doğru etikete çevir
kume_tahminleri_mapped = [kume_etiket_map[label] for label in kume_tahminleri]

# Performans metrikleri
accuracy = accuracy_score(etiketler_kodlu, kume_tahminleri_mapped)
precision = precision_score(etiketler_kodlu, kume_tahminleri_mapped, average="weighted", zero_division=0)
recall = recall_score(etiketler_kodlu, kume_tahminleri_mapped, average="weighted", zero_division=0)
f1 = f1_score(etiketler_kodlu, kume_tahminleri_mapped, average="weighted", zero_division=0)
conf_matrix = confusion_matrix(etiketler_kodlu, kume_tahminleri_mapped)

# Sonuçları yazdırma
print("K-Means Accuracy:", accuracy)
print("K-Means Precision:", precision)
print("K-Means Recall:", recall)
print("K-Means F1-Score:", f1)
print("K-Means Confusion Matrix:\n", conf_matrix)
print("K-Means Classification Report:\n", classification_report(etiketler_kodlu, kume_tahminleri_mapped, target_names=unique_labels, zero_division=0))

# PCA ile görselleştirme
pca = PCA(n_components=2)
v_reduced = pca.fit_transform(v.toarray())  # TF-IDF matrisini numpy array'e çevirip PCA uygula
plt.figure(figsize=(10, 8))
for i, label in enumerate(unique_labels):
    plt.scatter(v_reduced[kume_tahminleri == i, 0], v_reduced[kume_tahminleri == i, 1], s=50, label=f"Kume {label}")
plt.title("Emotion Veri Kümesi - K-Means Algoritması Görseli")
plt.legend()
plt.show()