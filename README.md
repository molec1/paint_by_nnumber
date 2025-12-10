# Paint-By-Numbers Generator

A Python tool that converts photos or illustrations into **print-ready paint-by-numbers worksheets** (outline page + palette + booklet).

✔ Supports A4 / A3 / A2 print scales  
✔ Includes smoothing, region merging, palette generation, numbering and PDF export  
✔ Output includes:
- Quantized preview
- Clean outline with numbered regions
- Colored reference version
- Palette image + CSV
- A 2-page PDF booklet

Designed for artists, print shops, coloring-book creators, or hobbyists.

---

## ✨ Features

- Automatic color quantization in CIE Lab space
- Region segmentation + connected component merging
- Physical-scale–aware region filtering  
  (small regions invisible in print are merged automatically)
- Smart placement of region numbers
- PDF booklet with:
  - Full-size coloring sheet  
  - Original image + miniature coloring preview  
  - Full color palette with names

---

## 📸 Input → Output Example

Input image:


