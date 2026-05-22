#include "utils.h"
#include <iostream>
#include <chrono>
#include <thread>
#include <cmath>

using namespace std;
using Clock = chrono::high_resolution_clock;
 
int get_thread_count() {
    unsigned t = thread::hardware_concurrency();   
    return t == 0 ? 4 : static_cast<int>(t);       
}

// Assure que deux images ont la même taille en redimensionnant la seconde
// Si les tailles diffèrent, redimensionne B pour qu'elle corresponde à A
void ensure_same_size(cv::Mat &A, cv::Mat &B) {
    if (A.size() != B.size()) {
        // Redimensionne B à la taille de A avec interpolation linéaire
        cv::resize(B, B, A.size(), 0, 0, cv::INTER_LINEAR);
    }
}

// Convertit une image couleur BGR en niveaux de gris
// Retourne une nouvelle image monochrome
cv::Mat convert_to_grayscale(const cv::Mat &color) {
    cv::Mat gray;
    cv::cvtColor(color, gray, cv::COLOR_BGR2GRAY);  // Conversion standard BGR → niveaux de gris
    return gray;
}

// Calcule le coefficient de Bhattacharyya entre deux histogrammes normalisés
// Mesure la similarité entre deux distributions de probabilités
// Valeur de retour : 0.0 (complètement différents) à 1.0 (identiques)
double bhattacharyya_coeff(const vector<float> &h1, const vector<float> &h2) {
    double bc = 0.0;  
    size_t n = h1.size();  // Nombre de bins dans les histogrammes
    
    // Parcourt tous les bins des histogrammes
    for (size_t i = 0; i < n; ++i) {
        // Assure que les valeurs sont non négatives (protection contre les erreurs d'arrondi)
        double a = max(0.0, static_cast<double>(h1[i]));  // Probabilité dans le bin i de h1
        double b = max(0.0, static_cast<double>(h2[i]));  // Probabilité dans le bin i de h2
        
        // Formule de Bhattacharyya : somme des racines carrées des produits
        bc += sqrt(a * b);  // √(p_i × q_i)
    }
    
   
    bc = max(0.0, min(1.0, bc)); 
    return bc;
}

 