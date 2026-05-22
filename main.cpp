#include "color_comparator.h"
#include "gradient_comparator.h"
#include "texture_comparator.h"
#include "utils.h"
#include <iostream>
#include <iomanip>
#include <memory>

int main(int argc, char** argv) {
    if (argc < 3) {
        std::cerr << "Usage: " << argv[0] << " imageA imageB [hist_bins]\n";
        std::cerr << "Exemple: " << argv[0] << " chat.jpg chien.jpg 64\n";
        return 1;
    }
    
    // Configuration
    int histBins = (argc >= 4) ? std::atoi(argv[3]) : 64;
    if (histBins < 16 || histBins > 256) {
        std::cerr << "hist_bins doit être entre 16 et 256, utilisation de 64\n";
        histBins = 64;
    }
    
    // Chargement des images
    cv::Mat A = cv::imread(argv[1], cv::IMREAD_COLOR);
    cv::Mat B = cv::imread(argv[2], cv::IMREAD_COLOR);
    
    if (A.empty() || B.empty()) {
        std::cerr << "Échec du chargement des images\n";
        return 1;
    }
    
    // Assurer que les images ont la même taille
    ensure_same_size(A, B);
    
    std::cout << "Taille des images: " << A.cols << "x" << A.rows << "\n";
    std::cout << "Utilisation de " << get_thread_count() << " threads\n";
    std::cout << "Bins d'histogramme: " << histBins << "\n\n";
    
    // Création des comparateurs
    ColorComparator colorComp(16, histBins);
    GradientComparator gradientComp(3);
    TextureComparator textureComp(16, std::min(256, histBins));
    
    // Conversion en niveaux de gris (une seule fois)
    cv::Mat grayA, grayB;
    cv::cvtColor(A, grayA, cv::COLOR_BGR2GRAY);
    cv::cvtColor(B, grayB, cv::COLOR_BGR2GRAY);
    
    // Timer pour l'exécution totale
    Timer totalTimer;
    double colorTime = 0, gradientTime = 0, textureTime = 0;
    
    // Exécution des comparaisons avec timing individuel
    Timer colorTimer;
    double colorSim = colorComp.compare(A, B);
    colorTime = colorTimer.elapsed_ms();
    
    Timer gradientTimer;
    double gradSim = gradientComp.compare(grayA, grayB);
    gradientTime = gradientTimer.elapsed_ms();
    
    Timer textureTimer;
    double textureSim = textureComp.compare(grayA, grayB);
    textureTime = textureTimer.elapsed_ms();
    
    // Obtenir le temps total
    double totalTime = totalTimer.elapsed_ms();
    
    // Combinaison pondérée
    const double WEIGHT_COLOR = 0.4;
    const double WEIGHT_GRAD = 0.3;
    const double WEIGHT_TEXTURE = 0.3;
    
    double combinedSim = WEIGHT_COLOR * colorSim +
                         WEIGHT_GRAD * gradSim +
                         WEIGHT_TEXTURE * textureSim;
    
    combinedSim = std::min(1.0, std::max(0.0, combinedSim));
    double diffPercent = (1.0 - combinedSim) * 100.0;
    
    // Affichage des résultats
    std::cout << "\n" << std::string(40, '=') << "\n";
    std::cout << "CHRONOMÉTRAGE DES PERFORMANCES:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::fixed << std::setprecision(2);
    std::cout << "Comparaison couleur:    " << std::setw(6) << colorTime << " ms\n";
    std::cout << "Comparaison gradient:   " << std::setw(6) << gradientTime << " ms\n";
    std::cout << "Comparaison texture:    " << std::setw(6) << textureTime << " ms\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << "Calcul total:           " << std::setw(6) << totalTime << " ms\n\n";
    
    std::cout << "\n" << std::string(40, '=') << "\n";
    std::cout << "RÉSULTATS DE SIMILARITÉ:\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << std::fixed << std::setprecision(3);
    std::cout << "Similarité couleur:     " << std::setw(6) << colorSim * 100 << "%\n";
    std::cout << "Similarité gradient:    " << std::setw(6) << gradSim * 100 << "%\n";
    std::cout << "Similarité texture:     " << std::setw(6) << textureSim * 100 << "%\n";
    std::cout << std::string(40, '-') << "\n";
    std::cout << "Similarité combinée:    " << std::setw(6) << combinedSim * 100 << "%\n";
    std::cout << "Différence:             " << std::setw(6) << diffPercent << "%\n";
    std::cout << std::string(40, '=') << "\n\n";
    
    // Détection de rotation possible (une seule ligne)
    if (textureSim > 0.7 && gradSim < 0.5) {
        std::cout << " NOTE: Texture élevée mais couleurs/gradients faibles - images possiblement similaires mais tournées\n\n";
    }
    
    // Résumé des performances
    std::cout << "RÉSUMÉ DES PERFORMANCES:\n";
    std::cout << "Temps total: " << totalTime << " ms (" << (totalTime/1000.0) << " secondes)\n";
    std::cout << "Threads utilisés: " << get_thread_count() << "\n";
    
    return 0;
}