import random
import string
import PIL
from PIL import ImageFont, Image, ImageDraw
import numpy as np
import cv2
 
font_list, imgnum = ['arial', 'arialbd', 'arialbi', 'ariali'], 10

for i in range(imgnum):
    font = ImageFont.truetype(random.choice(font_list)+".ttf",14)
    img = Image.new("RGBA", (110,20),(255,255,255))
    word = ''.join(random.choice(string.ascii_letters) for i in range(random.randrange(5,10)))
    ImageDraw.Draw(img).text((5, 0), word, (0,0,0), font=font)
    img.save('english/img/'+word+".png")
    img = cv2.imread('english/img/'+word+".png")
    cv2.imwrite('english/noise/'+word+"_n.png", img + img * np.random.normal(0,1,img.size).reshape(img.shape[0],img.shape[1],img.shape[2]).astype('uint8'))
    #with open('english/txt/'+word+'.txt', 'w', encoding = 'utf8') as txt_file: txt_file.write(word)
