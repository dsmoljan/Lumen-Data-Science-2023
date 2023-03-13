
[https://paperswithcode.com/task/audio-classification](https://paperswithcode.com/task/audio-classification)

  

[BEATs: Audio Pre-Training with Acoustic Tokenizers](https://arxiv.org/pdf/2212.09058v1.pdf)

-   ovaj bi se možda mogao koristiti i kao feature extractor, tj. možda bi njega mogli fine-tunati na našem tasku?
    
-   ovaj pristup baš koristi čisti audio
    
-   [https://github.com/microsoft/unilm/tree/master/beats](https://github.com/microsoft/unilm/tree/master/beats) --> kod
    

  

  

[https://developers.google.com/learn/pathways/get-started-audio-classification](https://developers.google.com/learn/pathways/get-started-audio-classification)

-   ovaj googleov tutorial baš navodi da je klasifikacija spektograma (kao slika) čest pristup
    

  

[https://shareg.pt/I2gaECp](https://shareg.pt/I2gaECp)

-   malo korisne teorije o audio feature extractionu
    
-   mogli bi napraviti neke od ovih ekstrakcija featurea i onda to klasificirati nekim modelom
    
-   ali i dalje mi se čini bolje koristiti onaj BEATs model koji radi ekstrakciju featurea i samo njega fine-tuneati
    

  

[Music and Instrument Classification using DeepLearning Technics](https://cs230.stanford.edu/projects_fall_2019/reports/26225883.pdf)

-   neki studentski projekt s Stanforda koji se bavio istom stvari
    
-   oni rade image classification nad mel-spectogramom
    
-   spominju da je dobro dodati augmentaciju (npr. white noise) uzorcima da se izbjegne overfitting
    

  

[Musical Instrument Identification Using Deep Learning Approach](https://www.ncbi.nlm.nih.gov/pmc/articles/PMC9025072/)

-   jedan malo ozbiljniji paper koji se isto bavi klasifikacijom instrumenata
    
-   uključuje zanimljivu meta-analizu drugih sličnih radova, pristupa koji su koristili i koje su rezultate dobili
    
-   koriste MFCC
    

[Jazz Solo Instrument Classification](https://archives.ismir.net/ismir2018/paper/000145.pdf)

-   ovaj rad je zanimljiv jer koriste još neke metode za izvlačenje specifičnih featura iz uzorka prije nego što uzorak ubace u neuronku
    
-   isti dataset koji mi imamo
    

  

[Predominant Instrument Recognition in PolyphonicMusic Using Convolutional Recurrent Neural Networks](https://cmmr2021.github.io/proceedings/pdffiles/cmmr2021_21.pdf)

-   ovaj koristi mel spektograme + još nešto kako bi dobio bolje rezultate
    

-   tj. samo radi fuziju predikcije dobivene pomoću spektograma i predikcije dobivene pomoću tog nečeg drugog
    

-   također koristi waveGAN kako bi generirao još podataka
    

  

[Efficient Training of Audio Transformers with Patchout](https://arxiv.org/pdf/2110.05069v3.pdf)

-   ovi treniraju transformersku arhitekturu nad mel spektogramima
    
-   modifikacija transformerse arhitekture koja zahtijeva nešto manje resursa i realno ju je moguće od početka trenirati na consumer GPU
    
-   zanimljiv rad jer je de-facto ono što mi planiramo, osim drugog dataseta (oni koriste Audioset)
    
-   također, spominju neke potencijalno korisne metod augmentacije (!)
    

  

[IRMAS classification - Feature Extraction + ML (not DL!)](https://github.com/AbubakarSarwar/Instrument-Classification-of-IRMAS-Dataset)

-   potencijalno koristan repo
    
-   čovjek je extractao 8 različitih featurea iz svakog od uzoraka u IRMAS datasetu i onda pomoću tih featurea radio klasifikaciju klasičnim ML metodama
    
-   ovo može biti jedna od metoda u ansamblima
    
-   i također možemo implementirati i usporediti rad s glavnim modelom
    

  

[https://github.com/micmarty/Instronizer](https://github.com/micmarty/Instronizer)

-   zanimljivo zbog činjenice da tvrde da je IRMAS jako loš dataset -> da ima jako puno krivo označenih primjera, te da su primjeri iz malog skupa pjesama
    
-   također da je nebalansiran po instrumentima

Bitna napomena: uskoro stiže PyTorch 2.0, koji uvodi model = torch.compile(model) funkcionalnost, koja može znatno ubrzati treniranje -> stoga ne zaboraviti to implementirati!
