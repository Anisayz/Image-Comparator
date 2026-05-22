#include "color_comparator.h"
#include "utils.h"
#include <thread>
#include <vector>
#include <algorithm>

using namespace std;
using namespace cv;

// Constructeur - initialise les paramètres de comparaison des couleurs
ColorComparator::ColorComparator(int blocks, int histBins) 
    : blocks_(blocks), histBins_(histBins) {
}

// Calcule l'histogramme de couleur pour un bloc spécifique de l'image
void ColorComparator::compute_block_histogram(const Mat &img, int bx, int by,
                                              vector<vector<float>> &hist) const {
    int rows = img.rows, cols = img.cols;
    int block_w = cols / blocks_;      
    int block_h = rows / blocks_;      
    
    // Calcule les coordonnées du bloc dans l'image
    int x0 = bx * block_w;             
    int y0 = by * block_h;              
    int x1 = (bx == blocks_ - 1) ? cols : (x0 + block_w);   
    int y1 = (by == blocks_ - 1) ? rows : (y0 + block_h);  
    
    // Initialise les histogrammes : 3 canaux (R, G, B) × histBins_ bins
    hist.assign(3, vector<float>(histBins_, 0.0f));
    int cnt = 0;  // Compteur de pixels dans le bloc
    
    // Parcourt tous les pixels du bloc
    for (int r = y0; r < y1; ++r) {
        const Vec3b* prow = img.ptr<Vec3b>(r);  
        for (int c = x0; c < x1; ++c) {
            const Vec3b &v = prow[c];   
        
            // Quantification des couleurs : mappe 0-255 → 0-(histBins_-1)
            // >> 8 est équivalent à /256 (division rapide par bit shifting)
            int br = (v[2] * histBins_) >> 8;  
            int bg = (v[1] * histBins_) >> 8;  
            int bb = (v[0] * histBins_) >> 8;  
            
            // Incrémente les bins correspondants
            hist[0][br] += 1.0f;   
            hist[1][bg] += 1.0f;  
            hist[2][bb] += 1.0f; 
            ++cnt;
        }
    }
    
    // Normalisation : convertit les comptes en probabilités (somme = 1.0)
    if (cnt > 0) {
        float total = static_cast<float>(cnt);
        for (int ch = 0; ch < 3; ++ch) {
            for (int b = 0; b < histBins_; ++b) {
                hist[ch][b] /= total;
            }
        }
    }
}

// Calcule les histogrammes pour TOUS les blocs de l'image (multi-threadé)
void ColorComparator::compute_all_histograms(const Mat &img,
                                             vector<vector<vector<float>>> &allHists) const {
    // Alloue la mémoire pour tous les blocs : blocks_² blocs × 3 canaux × histBins_ bins
    allHists.assign(blocks_ * blocks_, vector<vector<float>>(3, vector<float>(histBins_, 0.0f)));
    int nthreads = get_thread_count();   
    
    // Fonction de travail exécutée par chaque thread
    auto worker = [&](int tid) {
        int totalBlocks = blocks_ * blocks_;
        // Calcule la plage de blocs assignée à ce thread
        int b0 = (totalBlocks * tid) / nthreads;       
        int b1 = (totalBlocks * (tid + 1)) / nthreads; 
        if (tid == nthreads - 1) b1 = totalBlocks; 
        
        // Traite chaque bloc dans la plage assignée
        for (int bi = b0; bi < b1; ++bi) {
            int bx = bi % blocks_;  // Coordonnée X du bloc
            int by = bi / blocks_;  // Coordonnée Y du bloc
            vector<vector<float>> hist;
            compute_block_histogram(img, bx, by, hist);  // Calcule l'histogramme du bloc
            allHists[bi] = move(hist);  // Transfert sans copie (optimisation)
        }
    };
    
    // Crée et lance les threads
    vector<thread> threads;
    threads.reserve(nthreads);  // Réserve l'espace mémoire
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);  // Crée et démarre le thread
    }
    
    // Attend que tous les threads terminent
    for (auto &t : threads) {
        t.join();
    }
}

// Compare deux images couleur et retourne un score de similarité (0.0 à 1.0)
double ColorComparator::compare(const Mat &imgA, const Mat &imgB) {
    
    Mat A = imgA.clone();
    Mat B = imgB.clone();
    ensure_same_size(A, B);
    
    // Calcule les histogrammes pour les deux images (en parallèle)
    vector<vector<vector<float>>> histsA, histsB;
    compute_all_histograms(A, histsA);
    compute_all_histograms(B, histsB);
    
    int totalBlocks = static_cast<int>(histsA.size());
    double sumSim = 0.0;
    blockSimilarities_.resize(totalBlocks);  // Stocke les similarités par bloc
    
    // Compare chaque bloc correspondant entre les deux images
    for (int bi = 0; bi < totalBlocks; ++bi) {
        double perBlock = 0.0;
        // Compare les 3 canaux de couleur séparément
        for (int ch = 0; ch < 3; ++ch) {
            // Coefficient de Bhattacharyya : mesure la similarité entre deux distributions
            double bc = bhattacharyya_coeff(histsA[bi][ch], histsB[bi][ch]);
            perBlock += bc;
        }
        perBlock /= 3.0;  // Moyenne des similarités Rouge + Vert + Bleu
        blockSimilarities_[bi] = perBlock;  // Stocke la similarité de ce bloc
        sumSim += perBlock;  
    }
    
 
    return sumSim / static_cast<double>(totalBlocks);
}