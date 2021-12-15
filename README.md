# Summary

This repository is an implementation of a CNN for use in image preprocessing to improve OCR accuracy. It borrows code from the GitHub repo ["FSRCNN-pytorch"](https://github.com/yjn870/FSRCNN-pytorch) (coupled with paper ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367)).

## Implementation & Model Details

This model implements an FSRCNN that attempts to improve OCR accuracy through image preprocessing by modifying images to remove their noise and make them look like the original image. The model uses MSE to calculate the difference between the two images. Weighted Levenshtein distance was originally used to improve OCR, but training yielded poor results as it was too vague a metric to yield useful weight updates. 

## Dependencies

- pytorch
- numpy
- pillow
- pickle
- tqdm
- pytesseract
- weighted_levenshtein
- cv2
- glob

## Datasets

`prepare.py` is used to create random images, insert noise into them, and create datasets on which to train and evaluate.
Use the following command to do so
```bash
python prepare.py --train-num 1048576 --eval-num 2048 --train-path dataset/train2_20.pickle --eval-path dataset/eval2_11.pickle
```
where ```train-num``` is the number of image-noisy image pairs in the training set and ```eval-num``` is the number of image-noisy image pairs in the evaluation set.

## Train
To train, use the following command and specify the train and eval files

```bash
python train.py --train-file dataset/train2_20.pickle --eval-file dataset/eval2_11.pickle --outputs-dir output --batch-size 4 --num-epochs 40000000
```
Note that training takes a lot of time and you will not see noticeable results for tens, hundreds, or thousands of epochs, depending on the size of the image.
