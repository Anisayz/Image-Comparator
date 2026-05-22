# Projet de Comparaison d'Images

## Description
Outil de comparaison d'images en C++ utilisant trois méthodes complémentaires :
- **Couleur** : Analyse par blocs avec histogrammes RGB
- **Gradient** : Détection de contours avec opérateur Sobel  
- **Texture** : Analyse de motifs avec LBP (Local Binary Patterns)

## Compilation

### Prérequis
- Compilateur C++17 (g++, clang++, MSVC)
- OpenCV 4.x
- CMake (optionnel)

### Compilation manuelle (MSYS2/Mingw)
```bash
g++ -O3 -std=c++17 -I/mingw64/include/opencv4 -L/mingw64/lib \
    main.cpp color_comparator.cpp gradient_comparator.cpp \
    texture_comparator.cpp utils.cpp \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
    -o imgdiff.exe

## Execution : 

./imgdiff.exe image1.jpg image2.jpg [bins]

# Exemples :
./imgdiff.exe chat.jpg chat2.jpg 64
./imgdiff.exe paysage1.png paysage2.png