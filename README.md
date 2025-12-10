Paint-By-Numbers Generator

This project turns a photo into a paint-by-numbers worksheet.

It produces:
• a clean outline
• a colored reference image
• a color palette (image + CSV)
• a PDF booklet with all pages

It supports printing formats A4, A3 and A2.

Main Features
• Color reduction using K-Means in Lab space
• Detection and merging of regions
• Minimum region size depends on paper size
• Smart placement of color numbers inside shapes
• Ready-to-print PDF output

Usage

Basic example:
python main.py photo.jpg


With paper size:
python main.py photo.jpg A2


Supported values:
A4, A3, A2

Output Files

Generated files are placed in the output/ folder:
output/photo_quantized.png
output/photo_pbn_outline.png
output/photo_pbn_colored.png
output/photo_palette.csv
output/photo_palette.png
output/photo_booklet.pdf

How It Works
• Load the image
• Quantize colors
• Smooth and clean the color map
• Detect connected regions
• Merge regions that are too small
• Measure region centers
• Place numbers inside shapes
• Build palette files
• Render final images
• Create a 2-page booklet PDF

Project Structure
pbn-generator/
    main.py
    pipeline.py
    config.py
    quantization.py
    smoothing.py
    segmentation.py
    palette_utils.py
    rendering.py
    pdf_booklet.py
    output/

Status

Working prototype — future improvements planned:

• configuration file
• palette brightness options
• PDF scaling controls
