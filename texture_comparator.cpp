#include "texture_comparator.h"
#include "utils.h"
#include <thread>
#include <vector>

using namespace std;
using namespace cv;

// Constructeur - initialise les paramètres d'analyse de texture
TextureComparator::TextureComparator(int blocks, int histBins) 
    : blocks_(blocks), histBins_(histBins) {
}

// Calcule la carte LBP (Local Binary Patterns) d'une image en niveaux de gris
void TextureComparator::compute_lbp(const Mat &gray, Mat &lbp) const {
    CV_Assert(gray.type() == CV_8UC1);  // Vérifie que l'image est en niveaux de gris
    
    // Initialise la carte LBP (même taille que l'original, valeurs 0-255)
    lbp = Mat::zeros(gray.size(), CV_8UC1);
    int rows = gray.rows;
    int cols = gray.cols;
    int nthreads = get_thread_count();  
    
   
    auto worker = [&](int tid) {
        // Calcule les lignes à traiter pour ce thread (exclut les bords de 1 pixel)
        int innerRows = max(0, rows - 2);  // Lignes internes (sans bords)
        int r0 = 1 + (innerRows * tid) / nthreads;      // Ligne de début
        int r1 = 1 + (innerRows * (tid + 1)) / nthreads; // Ligne de fin
        if (tid == nthreads - 1) r1 = rows - 1;         // Dernier thread prend les lignes restantes
        
        // Traite chaque ligne assignée à ce thread
        for (int r = r0; r < r1; ++r) {
            // Pointeurs vers les 3 lignes du voisinage 3x3
            const uchar* prev = gray.ptr<uchar>(r - 1);  // Ligne du dessus
            const uchar* curr = gray.ptr<uchar>(r);      // Ligne courante (centre)
            const uchar* next = gray.ptr<uchar>(r + 1);  // Ligne du dessous
            uchar* dst = lbp.ptr<uchar>(r);              // Ligne de sortie LBP
            
            // Parcourt chaque colonne (exclut les bords de 1 pixel)
            for (int c = 1; c < cols - 1; ++c) {
                uchar center = curr[c];   
                uchar code = 0;           
                
                // Compare chaque voisin avec le pixel central (sens horaire)
                // Si voisin ≥ centre → bit = 1, sinon bit = 0
                code |= (prev[c - 1] >= center) << 7;  // Haut-gauche (bit 7)
                code |= (prev[c] >= center) << 6;      // Haut (bit 6)
                code |= (prev[c + 1] >= center) << 5;  // Haut-droite (bit 5)
                code |= (curr[c + 1] >= center) << 4;  // Droite (bit 4)
                code |= (next[c + 1] >= center) << 3;  // Bas-droite (bit 3)
                code |= (next[c] >= center) << 2;      // Bas (bit 2)
                code |= (next[c - 1] >= center) << 1;  // Bas-gauche (bit 1)
                code |= (curr[c - 1] >= center) << 0;  // Gauche (bit 0)
                
                dst[c] = code;  // Stocke le code LBP (0-255)
            }
        }
    };
    
    // Crée et lance les threads
    vector<thread> threads;
    threads.reserve(nthreads);   
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);   
    }
    
   
    for (auto &t : threads) {
        t.join();
    }
}

// Calcule l'histogramme LBP pour un bloc spécifique
void TextureComparator::compute_block_lbp_histogram(const Mat &lbp, int bx, int by,
                                                    vector<float> &hist) const {
    int rows = lbp.rows;
    int cols = lbp.cols;
    int block_w = cols / blocks_;  
    int block_h = rows / blocks_;  
    
    // Calcule les coordonnées du bloc dans l'image LBP
    int x0 = bx * block_w;            
    int y0 = by * block_h;            
    int x1 = (bx == blocks_ - 1) ? cols : (x0 + block_w);  
    int y1 = (by == blocks_ - 1) ? rows : (y0 + block_h);  
    
    // Initialise l'histogramme avec histBins_ bins  
    hist.assign(histBins_, 0.0f);
    int cnt = 0;  // Compteur de pixels dans le bloc
    
    // Parcourt tous les pixels du bloc dans la carte LBP
    for (int r = y0; r < y1; ++r) {
        const uchar* prow = lbp.ptr<uchar>(r);  
        for (int c = x0; c < x1; ++c) {
            int code = prow[c];  
           
            int bin = (code * histBins_) >> 8;  // >>8 = division par 256
            hist[bin] += 1.0f;  
            ++cnt;
        }
    }
    
    // Normalisation : convertit les comptes en probabilités
    if (cnt > 0) {
        float total = static_cast<float>(cnt);
        for (float &val : hist) {
            val /= total;  // Chaque bin = proportion de pixels avec ce pattern
        }
    }
}

// Calcule les histogrammes LBP pour TOUS les blocs (multi-threadé)
void TextureComparator::compute_all_lbp_histograms(const Mat &lbp,
                                                   vector<vector<float>> &allHists) const {
    // Alloue la mémoire : blocks_² blocs × histBins_ bins
    allHists.assign(blocks_ * blocks_, vector<float>(histBins_, 0.0f));
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
            int bx = bi % blocks_;   
            int by = bi / blocks_;   
            vector<float> hist;
            compute_block_lbp_histogram(lbp, bx, by, hist);   
            allHists[bi] = move(hist);  // Transfert sans copie (optimisation)
        }
    };
    
   
    vector<thread> threads;
    threads.reserve(nthreads);
    for (int t = 0; t < nthreads; ++t) {
        threads.emplace_back(worker, t);
    }
    
   
    for (auto &t : threads) {
        t.join();
    }
}

// Compare la texture de deux images et retourne un score de similarité (0.0 à 1.0)
double TextureComparator::compare(const Mat &grayA, const Mat &grayB) {
 
    CV_Assert(grayA.type() == CV_8UC1 && grayB.type() == CV_8UC1);
    CV_Assert(grayA.size() == grayB.size());
    
    // Étape 1 : Calcul des cartes LBP pour les deux images
    Mat lbpA, lbpB;
    compute_lbp(grayA, lbpA);   
    compute_lbp(grayB, lbpB);   
    
    // Étape 2 : Calcul des histogrammes LBP par blocs
    vector<vector<float>> histsA, histsB;
    compute_all_lbp_histograms(lbpA, histsA);  
    compute_all_lbp_histograms(lbpB, histsB);  
    
    // Étape 3 : Comparaison des histogrammes bloc par bloc
    int totalBlocks = static_cast<int>(histsA.size()); 
    double sumSim = 0.0;  
    
    // Pour chaque bloc correspondant dans les deux images
    for (int bi = 0; bi < totalBlocks; ++bi) {
        // Coefficient de Bhattacharyya : mesure la similarité entre distributions
        double bc = bhattacharyya_coeff(histsA[bi], histsB[bi]);
        sumSim += bc;   
    }
    
    // Étape 4 : Retourne la moyenne des similarités sur tous les blocs
    return sumSim / static_cast<double>(totalBlocks);
}