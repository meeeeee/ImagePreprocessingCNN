import random
import string
import PIL
from PIL import ImageFont
from PIL import Image
from PIL import ImageDraw
from tqdm import tqdm
import numpy as np
import cv2
 
# create a list of characters to be used in creating dataset
char_list = []
for char in string.ascii_letters:
    char_list.append(char)
    
# create font list
font_lst = ['arial', 'arialbd', 'arialbi', 'ariali']  #, 'times', 'timesbd', 'timesi','ariblk', 'arialbd', 'timesbi'
 
# generate images for each fonts
imgnum = 10
for fonts in font_lst:
    for i in tqdm(range(imgnum)):
        for i in range(len(char_list)):
            # Choose a random word size
            word_size = random.randrange(5,10)
            # create word starting with the current character
            char_list_copy = char_list.copy()
            char_list_copy.remove(char_list[i])
            new_word = char_list[i]
            for _ in range(word_size): new_word += random.choice(char_list_copy)
            # Draw the word on the image
            font = ImageFont.truetype(fonts+".ttf",14)
            img=Image.new("RGBA", (110,20),(255,255,255))
            draw = ImageDraw.Draw(img)
            draw.text((5, 0),new_word,(0,0,0),font=font)
            # Save the image and the corresponding text file
            img.save('english/img/'+new_word+".png")
            img = cv2.imread('english/img/'+new_word+".png")
            cv2.imwrite('english/noise/'+new_word+"_n.png", img + img * np.ones(img.shape[0],img.shape[1],img.shape[2])*np.random.normal(0,1,img.size).reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8'))
            with open('english/txt/'+new_word+'.txt', 'w', encoding = 'utf8') as txt_file: txt_file.write(new_word)