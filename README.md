# Images Comparator

A C++ command-line tool that compares two images using three complementary analysis methods and produces a combined similarity score.

## How It Works

The comparison pipeline runs three independent analyses in parallel, then combines their scores with a weighted average:

| Method | Weight | Technique |
|---|---|---|
| **Color** | 40% | Block-based RGB histogram comparison (16×16 grid) |
| **Gradient** | 30% | Edge detection using the Sobel operator |
| **Texture** | 30% | Local Binary Patterns (LBP) histogram comparison |

The final output is a **combined similarity score** (0–100%) and a **difference percentage**, along with individual scores for each method and the time taken by each analysis.

A rotation hint is also printed when texture similarity is high but gradient similarity is low, suggesting the images may be visually similar but spatially rotated.

## Project Structure

```
.
├── main.cpp                  # Entry point: argument parsing, orchestration, output
├── color_comparator.h/.cpp   # Block-based RGB histogram comparison
├── gradient_comparator.h/.cpp# Sobel-based edge/gradient comparison
├── texture_comparator.h/.cpp # LBP-based texture comparison
└── utils.h/.cpp              # Image resizing, thread count, timer utilities
```

## Requirements

- C++17 compiler (g++, clang++, or MSVC)
- [OpenCV 4.x](https://opencv.org/)
- CMake (optional)

## Build

### MSYS2 / MinGW (Windows)

```bash
g++ -O3 -std=c++17 \
    -I/mingw64/include/opencv4 \
    -L/mingw64/lib \
    main.cpp color_comparator.cpp gradient_comparator.cpp \
    texture_comparator.cpp utils.cpp \
    -lopencv_core -lopencv_imgproc -lopencv_imgcodecs -lopencv_highgui \
    -o imgdiff.exe
```

### Linux / macOS

```bash
g++ -O3 -std=c++17 \
    $(pkg-config --cflags --libs opencv4) \
    main.cpp color_comparator.cpp gradient_comparator.cpp \
    texture_comparator.cpp utils.cpp \
    -o imgdiff
```

## Usage

```
./imgdiff imageA imageB [hist_bins]
```

- `imageA`, `imageB` — paths to the images to compare (JPEG, PNG, etc.)
- `hist_bins` *(optional)* — number of histogram bins, between 16 and 256 (default: **64**)

### Examples

```bash
./imgdiff.exe photo1.jpg photo2.jpg
./imgdiff.exe photo1.jpg photo2.jpg 128
./imgdiff landscape_a.png landscape_b.png 32
```

## Sample Output

```
Image size: 1920x1080
Using 8 threads
Histogram bins: 64

========================================
PERFORMANCE TIMING:
----------------------------------------
Color comparison:   12.34 ms
Gradient comparison:  8.91 ms
Texture comparison:   9.55 ms
----------------------------------------
Total computation:  30.80 ms

========================================
SIMILARITY RESULTS:
----------------------------------------
Color similarity:   87.500%
Gradient similarity: 74.200%
Texture similarity:  81.300%
----------------------------------------
Combined similarity: 81.890%
Difference:          18.110%
========================================
```

## Notes

- If the two images have different dimensions, they are automatically resized to match before comparison.
- Grayscale conversion (for gradient and texture analysis) is done once and shared between both comparators.
- The number of threads used is reported at startup and reflects the hardware concurrency available.
