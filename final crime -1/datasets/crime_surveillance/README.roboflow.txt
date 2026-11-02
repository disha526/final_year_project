# Urban Crime CCTV Detection Dataset - v4
https://universe.roboflow.com/surveillance-ai-research/urban-crime-cctv-detection/dataset/4

Provided by Surveillance AI Research Lab
License: CC BY 4.0

### Dataset Summary
- Total Images: 272
- Train Set: 190 images
- Valid Set: 54 images
- Test Set: 28 images
- Annotation Format: YOLOv8 / YOLO11 PyTorch TXT format
- Cameras: Authentic indoor & outdoor multi-angle CCTV surveillance cameras

### Classes (11):
0: Fighting
1: Assault
2: Robbery
3: Stealing
4: Shoplifting
5: Burglary
6: Shooting
7: Arson
8: Explosion
9: Vandalism
10: Normal activity

### Preprocessing & Augmentations Applied:
- Auto-Orient: Applied
- Resize: Auto-padded 640x640 for YOLO11 training
- CCTV Noise & Low-light Augmentations: Random contrast, camera sensor noise simulation
