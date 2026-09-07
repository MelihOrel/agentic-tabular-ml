# LinkedIn paylaşımı (taslak)

Yeni bir proje bitirdim: agentic-tabular-ml.

Excel dosyasını yükleyip temizleyen, model eğiten ve modeli servis eden bir platform. Buraya kadar tanıdık geliyor olabilir. Farklı olan taraf şu: yapay zeka asistanı bu işi kendi başına yapabiliyor ama veriye hiç dokunamıyor.

Asistanın elinde sadece on tane araç var. Veriyi görmüyor, sadece bu araçları çağırıp kısa cevaplar okuyor. Yaptığı her değişiklik "bunu asistan yaptı" etiketiyle kayda geçiyor ve tek tıkla geri alınabiliyor. Model eğitmek istediğinde önce bir kontrol katmanı devreye giriyor. Hedef sütun tek değerden mi oluşuyor, tarih sütununu mu tahmin etmeye çalışıyor, aslında kimlik numarası olan bir sütunu özellik mi sanmış, bunlara bakıp gerekirse eğitimi reddediyor ve nedenini yazıyor.

Hesaplanan sütun kısmında da eval kullanmadım. Yazdığınız formül önce ayrıştırılıyor ve sadece izin verilen işlemler çalıştırılıyor. Yani oraya sisteme zarar verecek bir şey yazmanın yolu yok.

Bir de şu var: Türkiye'den çıkan Excel dosyaları genelde sorunlu geliyor. Kodlama farklı, ayraç noktalı virgül, ondalık ayracı virgül. Pandas bunları sessizce yanlış okuyor ve Şükrü'yü bambaşka bir şeye çeviriyor. Bu yüzden dosyayı okuyan kısmı baştan yazdım ve neye göre karar verdiğini ekranda gösteriyor. Tahmin ediyor ama tahminini saklamıyor.

Model tarafında bilerek az model kullandım. Üç aile, düzgün çapraz doğrulama ve hiç görmediği bir test kümesi. On modelli sıralamalar çoğu zaman gerçek fark değil gürültü ölçüyor.

Test yazarken iki şey öğrendim, ikisini de README'ye yazdım. Birincisi, normallik testi büyük örneklemlerde gerçekten normal veriyi bile eliyor. Bu testin hatası değil, gücü. İkincisi, pandas 3 ile birlikte metin sütunlarını yakalamak için yıllardır kullandığımız kontrol sessizce çalışmayı bırakıyor. Bunu bir test yakaladı, yoksa fark etmeyecektim.

Kod, testler ve canlı demo profilimde. Geri bildirime açığım.

#veribilimi #makineogrenmesi #python #streamlit
