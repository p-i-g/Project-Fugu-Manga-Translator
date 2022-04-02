# Project-Fugu-Manga-Translator

Instructions:

Install CUDA version 11.2 and cuDNN version 8.1.0

Run the project in a new conda environment.

```
pip install tensorflow
pip install manga109api
pip install matplotlib
pip install opencv-python
pip install -U 'tensorflow-text==2.8.*'    
pip install -U scikit-image==0.16.2
pip install tensorflow-addons
pip install adabelief-tf
conda install -c conda-forge tesseract
conda install -c conda-forge pytesseract
```

Get Japanese tesseract from 

https://github.com/tesseract-ocr/tessdata/blob/main/jpn_vert.traineddata

copy into tessdata directory (.../Anaconda/envs/Project-Fugu-Manga-Translator/Library/bin/tessdata)

fugu is poisonous
