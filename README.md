# Iris Flower Classification (Level 1 - Project 1)
Bu proje, veri bilimi ve makine öğrenmesine giriş amacıyla geliştirilmiş, klasik Iris Çiçek Veri Seti kullanılarak tür tahmini yapan bir sınıflandırma uygulamasıdır. 3. sınıf bir bilgisayar mühendisliği öğrencisi olarak, ham veriden canlı tahmin sistemine kadar tüm süreç sıfırdan kodlanmıştır.

#Kazanılan Yetkinlikler
Bu projeyi geliştirirken şu teknik konuları deneyimledim:
Veri Yönetimi: Pandas kütüphanesi ile ham veriyi işleme ve DataFrame yapısına dönüştürme.
Keşifsel Veri Analizi (EDA): Seaborn ve Matplotlib kullanarak verilerin dağılımını görselleştirme ve türler arasındaki matematiksel ayrımı analiz etme.
Makine Öğrenmesi Hattı (Pipeline): Veriyi X (Features) ve y (Target) olarak ayırma, Train-Test Split ile modelin ezberlemesini (overfitting) engelleme.
Algoritma Uygulama: K-Nearest Neighbors (KNN) algoritmasını kullanarak sınıflandırma modeli oluşturma.
Yazılım Mimarisi: Kullanıcıdan anlık veri alan, hata yakalama (try-except) mekanizmalı bir canlı tahmin döngüsü kurma.

#Proje İçeriği
Proje klasörü şu temel bileşenlerden oluşmaktadır:
main.py: Veri yükleme, model eğitimi, görselleştirme ve canlı tahmin sistemini içeren ana kod dosyası.
Görselleştirme: Çanak yaprak (sepal) ve taç yaprak (petal) ölçülerine göre türlerin nasıl kümelendiğini gösteren scatter plot grafiği.
Canlı Tahmin: Kullanıcının terminal üzerinden girdiği ölçümlere dayanarak anlık tür tahmini yapan etkileşimli yapı.

#Kullanılan Teknolojiler
Dil: Python 3.14
Kütüphaneler: * Scikit-Learn (Model ve Veri Seti)
Pandas & NumPy (Veri İşleme)
Matplotlib & Seaborn (Görselleştirme)

#Örnek Çıktı
Model, test verileri üzerinde %100 doğruluk oranıyla çalışmaktadır.
Sepal Length: 5.1
Sepal Width: 3.5
Petal Length: 1.4
Petal Width: 0.2
>>> ANALİZ SONUCU: Bu çiçek bir SETOSA!

