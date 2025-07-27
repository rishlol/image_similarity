# image_similarity

**Multithreaded Image Similarity using OpenCV**

This tool takes an input image and find similar images using a hashing algorithm. User can specify an input image and either another image or a directory of images.

---

## Build Instructions
### Install Required Packages:
- opencv
- boost

---

### Windows:
```bash
cmake -DCMAKE_TOOLCHAIN_FILE="<vcpkg>\scripts\buildsystems\vcpkg.cmake" -B build -S . -G "Visual Studio 17"
```

### MacOS:
```bash
cmake -B build -S . -G Xcode
```

### Linux:
```bash
cmake -B build -S .
```

---

## Usage

### Compare two images
```bash
image_similarity <img> <img2>
```

### Find similar images from directory of images
```bash
image_similarity <img> <dir>
```