import pandas as pd
import seaborn as sns
import numpy as np
import matplotlib.pyplot as plt
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import accuracy_score

# 1. Veriyi Hazırla (Download Data)
iris = load_iris()
df = pd.DataFrame(data=iris.data, columns=iris.feature_names)

df['target'] = iris.target

X = iris.data   # Girdiler (Ölçümler)
y = iris.target  # Çıktılar (Etiketler: 0, 1, 2)
target_names = iris.target_names

print("Sistem Hazır! Veri setinin ilk 5 satırı:")
print(df.head())
print(f"\nTahmin edilecek türler: {target_names}")

# 2. Veriyi Böl (Train-Test Split)
# Önce bölmeliyiz ki eğitecek verimiz olsun
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)
print(f"Eğitim seti boyutu: {X_train.shape}")
print(f"Test seti boyutu: {X_test.shape}")

# 3. Modeli Oluştur ve Eğit (Model & Fit)
# Tahmin yapmadan ÖNCE modelin kim olduğunu ve neyi öğreneceğini söylemeliyiz
model = KNeighborsClassifier(n_neighbors=3)
model.fit(X_train, y_train)
print("\nModel başarıyla eğitildi!")

# 4. Tahmin Yap ve Başarıyı Ölç (Predict & Evaluate)
# Artık eğitilmiş bir modelimiz var, test verilerini sorabiliriz
y_pred = model.predict(X_test)
accuracy = accuracy_score(y_test, y_pred)
print(f"Modelin doğruluk oranı (Accuracy): %{accuracy * 100:.2f}")

# Veriyi görselleştirelim (Sepal length vs Sepal width)
plt.figure(figsize=(10, 6))
sns.scatterplot(data=df,  x='sepal length (cm)', y='sepal width (cm)',          #Verilerimizi 4 boyuttan 2 boyuta indirgeyerek
                hue='target', palette='viridis', s=100)                         #koordinat sistemine yerleştirme
#hue='target' ---> noktaları yerleştirirken çiçek türüne göre nokta rengini değiştir

plt.title("Iris Çiçek Türlerinin Dağılımı (Çanak Yaprak)")
plt.xlabel("Çanak Yaprak Uzunluğu (cm)")
plt.ylabel("Çanak Yaprak Genişliği (cm)")
plt.legend(title='Türler', labels=['Setosa', 'Versicolor', 'Virginica'])
plt.show()

print("\n" + "="*40)
print("   GELİŞMİŞ ÇİÇEK TAHMİN SİSTEMİ")
print("="*40)
print("Not: Çıkmak için ölçü yerine '0' giriniz.\n")

while True:
    try:
        #1. Kullanıcıdan tek tek ölçüleri alma
        s_len = float(input("Çanak Yaprak Uzunluğu (Sepal Length - örn: 5.1): "))
        if s_len == 0: break

        s_wid = float(input("Çanak Yaprak Genişliği (Sepal Width - örn: 3.5): "))
        p_len = float(input("Taç Yaprak Uzunluğu (Petal Length - örn: 1.4): "))
        p_wid = float(input("Taç Yaprak Genişliği (Petal Width - örn: 0.2): "))

        #2. Veriyi modelin anlayacağı formata (3b Dizi) haline getiriyoruz
        #Scikit-learn her zaman [[v1, v2, v3, v4]] şeklinde bir yapı bekler
        yeni_numune = np.array([[s_len, s_wid, p_len, p_wid]])

        #3. Modelden tahmin istiyoruz
        tahmin_index = model.predict(yeni_numune)[0]
        tahmin_ismi = target_names[tahmin_index]

        #4. Sonucu ekrana yazdırıyoruz
        print(f"\n>>> SONUÇ: Bu ölçülere göre çiçek bir: {tahmin_ismi.upper()} <<<\n")
        print("-" * 30)

    except ValueError:
        print("\n!! HATA: Lütfen sadaece sayısal değerler giriniz (Nokta kullanarak, örn: 5.5). \n")

print("\nProgram sonlandırıldı.İyi Günler!")