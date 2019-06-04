import cv2
import numpy as np
import sys
import re
from panorama import *

if __name__=="__main__":
    # image_names = sys.argv
    image_names = ["0", "0102_03.jpg","04.jpg"] #,"01.jpg","02.jpg"，注意命名，已经是合成的多图名字里要有下划线，下面不用resize处理
    if len(image_names) == 1:
        print("usage: python3 panorama_main.py image1 image2 image3 ...")
        exit()

    if len(image_names) == 2:
        print("usage: Plese give 2 or more images to this program.")
        exit()

    #image_names = [r'pic02\01.jpg', r'pic02\02.jpg', r'pic02\03.jpg']
    images = []
    panorama = []
    for i in range(1,len(image_names)):
        print( "Loading " + str(image_names[i]))
        img=cv2.imread(image_names[i],cv2.IMREAD_COLOR)
        if re.match(r"[\w\d]+_[\w\d]+.jpg",str(image_names[i])): #对合成的多图不进行resize_img，只是测试时用，尺寸已经调整
            images.append(Image(str(i), img))
            continue
        img = resize_image(img)
        images.append(Image(str(i), img))

    panorama.append(Image(images[0].name, images[0].image))

    print("Your images have been loaded. Generating panorama starts ...")
    for i in range(0,len(images)-1):
        panorama.append(Image(str(i+1),make_panorama(panorama[i],images[i+1],i))) #i作为计数，传给make_panorama的count

    print("A panorama image is generated.")
    cv2.imwrite("panorama_0123_04nopick.jpg",panorama[-1].image)
