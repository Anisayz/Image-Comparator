#include "gradient_comparator.h"
#include "utils.h"
#include <thread>
#include <vector>
#include <cmath>

using namespace std;
using namespace cv;

// Constructeur - initialise le noyau Sobel pour détecter les contours
GradientComparator::GradientComparator(int sobelKernel) 
    : sobelKernel_(sobelKernel) {
}

// Calcule la magnitude du gradient pour une image en niveaux de gris
Mat GradientComparator::compute_gradient_magnitude(const Mat &gray) const {
    CV_Assert(gray.type() == CV_8UC1);  // Vérifie que l'image est en niveaux de gris
    
    // Calcul des gradients horizontaux (Gx) et verticaux (Gy) avec l'opérateur Sobel
    Mat grad_x, grad_y;
    Sobel(gray, grad_x, CV_32F, 1, 0, sobelKernel_);  // Dérivée horizontale
    Sobel(gray, grad_y, CV_32F, 0, 1, sobelKernel_);  // Dérivée verticale
    
    // Matrice pour stocker la magnitude du gradient
    Mat magnitude(gray.size(), CV_32F);
    
    // Pour chaque pixel, calcule la magnitude = sqrt(Gx² + Gy²)
    for (int r = 0; r < gray.rows; ++r) {
        const float* px = grad_x.ptr<float>(r);  // Pointeur vers la ligne des gradients horizontaux
        const float* py = grad_y.ptr<float>(r);  // Pointeur vers la ligne des gradients verticaux
        float* pmag = magnitude.ptr<float>(r);   // Pointeur vers la ligne de sortie
        
        for (int c = 0; c < gray.cols; ++c) {
            // hypotf calcule sqrt(px[c]² + py[c]²) - magnitude du vecteur gradient
            pmag[c] = hypotf(px[c], py[c]);
        }
    }
    
    return magnitude;
}

// Compare les gradients de deux images et retourne un score de similarité (0.0 à 1.0)
double GradientComparator::compare(const Mat &grayA, const Mat &grayB, int nthreads) {
    // Vérifications de sécurité
    CV_Assert(grayA.type() == CV_8UC1 && grayB.type() == CV_8UC1);
    CV_Assert(grayA.size() == grayB.size());
    
    // Si aucun nombre de threads n'est spécifié, utilise tous les cœurs disponibles
    if (nthreads <= 0) {
        nthreads = get_thread_count();
    }
    
    // Calcul des magnitudes de gradient pour les deux images
    Mat magA = compute_gradient_magnitude(grayA);
    Mat magB = compute_gradient_magnitude(grayB);
    
    int rows = grayA.rows;
    int cols = grayA.cols;
    double totalPixels = static_cast<double>(rows * cols);  // Nombre total de pixels
    
    // Vecteur pour stocker les sommes partielles de chaque thread
    vector<double> partialSums(nthreads, 0.0);
    
    // Fonction de travail exécutée par chaque thread
    auto worker = [&](int tid) { 
        int r0 = (rows * tid) / nthreads;    
        int r1 = (rows * (tid + 1)) / nthreads;  
        if (tid == nthreads - 1) r1 = rows;     
        
        double sumLocal = 0.0;  
        
        // Parcourt les lignes assignées à ce thread
        for (int r = r0; r < r1; ++r) {
            const float* pA = magA.ptr<float>(r);   
            const float* pB = magB.ptr<float>(r);   
            
            // Parcourt chaque pixel de la ligne
            for (int c = 0; c < cols; ++c) {
               
                double diff = fabs(static_cast<double>(pA[c]) - static_cast<double>(pB[c]));
                
                sumLocal += 1.0 / (1.0 + diff);
            }
        }
        partialSums[tid] = sumLocal;  // Stocke la somme locale
    };
    
    // Crée et lance les threads
    vector<thread> threads;
    threads.reserve(nthreads);  
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);  
    }
    
    // Attend que tous les threads terminent
    for (auto &t : threads) {
        t.join();
    }
    
    // Combine les résultats de tous les threads
    double totalSum = 0.0;
    for (double sum : partialSums) {
        totalSum += sum;
    }
     
    return totalSum / totalPixels;
}

 
 