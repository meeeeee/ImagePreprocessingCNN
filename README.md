# FSRCNN

This repository is an implementation of a GAN for use in image preprocessing to improve OCR accuracy. It borrows code from the GitHub repo ["FSRCNN-pytorch"](https://github.com/yjn870/FSRCNN-pytorch) (coupled with paper ["Accelerating the Super-Resolution Convolutional Neural Network"](https://arxiv.org/abs/1608.00367)).

## Implementation & Model Details

This model implements a GAN with a fixed discriminator network that attempts to improve OCR accuracy through image preprocessing by modifying images before OCR is run on them. The structure of the GAN is as follows:
- Image is passed through FSRCNN
- OCR (PyTesseract) is run on output of FSRCNN --- this is the generator network
- Similarity of OCR output to ground truth text is computed using per-character weighted Levenshtein distance (weighted Levenshtein distance divided by average number of characters across both text) --- this is the discriminator network
- Generator trains on discriminator output

Architecturally, the model is implemented as follows:
- Image is passed through FSRCNN and output along with ground truth is used to train the FSRCNN
- The loss function on which the CNN trains takes in an image and the ground truth text, runs OCR on the image, and returns the per-character weighted Levenshtein distance between the OCR output and the ground truth

## Requirements

- PyTorch
- Numpy
- Pillow
- h5py
- tqdm
- PyTesseract
- weighted_levenshtein

## Datasets

`randomimggen.py` is used to create random images, insert noise into them, and create text files that contain the text in the images.
`WIP` is used to convert these images and text files into .h5 for use in training.
To just test the model, run `test.py` (EDIT FOR TESTING)

## Train
To train, use the following command and specify the train and eval files

```bash
python train.py --train-file "your_file" --eval-file "your_file" --outputs-dir "your_folder" --scale 1 --batch-size 32 --num-epochs 10 --seed your_number
```
