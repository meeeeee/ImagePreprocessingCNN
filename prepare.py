import argparse
import glob
import numpy as np
import PIL
import PIL.Image as pil_image
from PIL import ImageFont, Image, ImageDraw
import pickle
import random
import string
import cv2

parser = argparse.ArgumentParser()
parser.add_argument('--train-num', type=int, required=True)
parser.add_argument('--eval-num', type=int, required=True)
parser.add_argument('--train-path', type=str, required=True)
parser.add_argument('--eval-path', type=str, required=True)
args = parser.parse_args()
font_list = ['arial', 'arialbd', 'arialbi', 'ariali']
train_lst, eval_lst = list(), list()
font = ImageFont.truetype(random.choice(font_list)+".ttf",14)
for i in range(args.train_num):
    img = Image.new("RGBA", (110,20),(255,255,255))
    word = ''.join(random.choice(string.ascii_letters) for i in range(random.randrange(5,10)))
    ImageDraw.Draw(img).text((5, 0), word, (0,0,0), font=font)
    img = np.array(img.convert("L"), dtype=np.float32)
    train_lst.append((img, img + img * np.random.normal(0,1,img.size).reshape(img.shape[0], img.shape[1]).astype('uint8')))

for i in range(args.eval_num):
    img = Image.new("RGBA", (110,20),(255,255,255))
    word = ''.join(random.choice(string.ascii_letters) for i in range(random.randrange(5,10)))
    ImageDraw.Draw(img).text((5, 0), word, (0,0,0), font=font)
    img = np.array(img.convert("L"), dtype=np.float32)
    eval_lst.append((img, img + img * np.random.normal(0,1,img.size).reshape(img.shape[0], img.shape[1]).astype('uint8')))
out1, out2 = open(args.train_path, "wb"), open(args.eval_path, "wb")
pickle.dump(train_lst, out1), pickle.dump(eval_lst, out2)
out1.close(), out2.close()