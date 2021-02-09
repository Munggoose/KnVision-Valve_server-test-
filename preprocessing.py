import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import rotate
import os

def make_rectangle(img):
    height, width = img.shape[:2]
    res_list = []
    
    res1 = img[:height//2, :width//2]
    res1 = rotate(res1, angle=270)
    
    res2 = img[height//2:, :width//2]
    res2 = rotate(res2, angle=180)
    
    res3 = img[height//2:, width//2:]
    res3 = rotate(res3, angle=90)
    
    res4 = img[:height//2, width//2:]
    
    res_list= [res1, res2, res3, res4]
    return res_list

def find_circle_crop(img):

    height, width = img.shape[:2]
    
    img = img[int(height / 2 - 310): int(height / 2 + 310), int(width / 2 - 310 ) : int(width / 2  + 310)]
    height, width = img.shape[:2]

    mask = np.zeros((height, width))
    res = np.zeros((height, width, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                            1, 
                            100,
                            param1=50,param2=200,
                            minRadius=200, maxRadius=600)

    if circles is not None:
        circles = np.round(circles[0, :]).astype("int")
        for (x, y, r) in circles:
            cv2.circle(mask, (x, y), r, 255, cv2.FILLED)

            img[mask != 255] = 0

            mask=mask.astype('uint8')
            _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)

            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
 
            x,y,w,h = cv2.boundingRect(np.array(contours[0][0]))
 
            crop = img[y:y+h,x:x+w]
            return make_rectangle(crop)

#승찬

PREFIX_BMP = '.bmp'
PREFIX_PNG = '.png'

sav_iSize = 128

def prnFile(rootDir, prefix):
    files = os.listdir(rootDir)
    tmp = []
    for file in files:
        path = os.path.join(rootDir, file)
        if(file[-4:] == prefix):
            tmp.append(path)
    return tmp

def mask_circle_solid(pil_img, background_color, blur_radius, offset=0):
    background = Image.new(pil_img.mode, pil_img.size, background_color)

    offset = blur_radius * 2 + offset
    mask = Image.new("L", pil_img.size, 0)
    draw = ImageDraw.Draw(mask)
    draw.ellipse((offset, offset, pil_img.size[0] - offset, pil_img.size[1] - offset), fill=255)
    mask = mask.filter(ImageFilter.GaussianBlur(blur_radius))

    return Image.composite(pil_img, background, mask)

def img_Contrast(img):
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=1.5, tileGridSize=(8,8))
    cl = clahe.apply(l)

    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final

def get_preprocess_img(path):
    
    PREFIX_BMP = '.bmp'
    # PREFIX_PNG = '.png'
    imSize = 720
    imSizeX = 1280
    imSizeY = 720
    imCenter = 55+610/2
    xBias = 280
    yBias = 0
    
    filepath = path#prnFile(PATH, PREFIX_BMP)
    print(filepath)
    raw = cv2.imread(filepath, cv2.IMREAD_COLOR)
    img = raw
    img = raw[yBias:yBias+imSize, xBias:xBias+imSize]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                    cv2.HOUGH_GRADIENT, 1.5, 1270, param1 = 70, 
                    param2 = 40, minRadius = 315, maxRadius = 324)
    
    if detected_circles is not None: 
    
        # Convert the circle parameters a, b and r to integers. 
        detected_circles = np.uint16(np.around(detected_circles)) 
        for pt in detected_circles[0, :]: 
            a, b, r = pt[0], pt[1], pt[2] 
            # print(r)
            
            
            # cv2.circle(img, (a, b), r, (255,0,255), 1) 
            # cv2.imshow('hi', img)
            # cv2.waitKey(0)
                        
            a += xBias
            b += yBias
            # r += 10
            

            # cv2.rectangle(backGD, (0,0), (imSize, imSize), (255,255,255), -1)
            backGD_circle = np.zeros((imSizeY,imSizeX, 3), dtype="uint8")
            cv2.circle(backGD_circle, (a, b), r, (255,255,255), -1) 
            
            rectX = (a - r) 
            rectY = (b - r)

            raw = cv2.bitwise_and(backGD_circle, raw)
            img = raw[rectY:(rectY+2*r), rectX:(rectX+2*r)]
        
        return img 


def seperate_image(img):
    res_list = find_circle_crop(img)
    return res_list
