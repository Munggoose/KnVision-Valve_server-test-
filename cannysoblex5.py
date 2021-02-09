import cv2 
import numpy as np 
import os
import os.path
"""
        인풋이미지는 큰비젼 카메라로 찍은 것.

    Returns:
        sav_iSize 크기의 1구간 사진.
        1구간을 제외한 다른 부분은 검은색 마스킹 처리리
"""


PATH = 'C:\\Users\\Seungchan_HCI\\OneDrive - inha.edu\\HCI\\valve_seat\\정상\\'
PATH = 'C:\\Users\\Seungchan_HCI\\Desktop\\Nuts\\Nuts_preprocessed\\NORMAL\\'
# PATH = 'C:\\Users\\Seungchan_HCI\\Desktop\\Nuts\\Nuts_preprocessed\\dd\\'
PATH = 'C:\\Users\\Seungchan_HCI\\Desktop\\Nuts\\Nuts_preprocessed\\2\\0208\\purified\\normal\\'
PATH = 'C:\\Users\\Seungchan_HCI\\Desktop\\Nuts\\Nuts_preprocessed\\2\\0208\\purified\\abnormal\\'

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

def get_preprocess_img(PATH):
    imsize - 720
    imSizeX = 1280
    imSizeY = 720
    imCenter = 55+610/2
    xBias = 280
    yBias = 0
    
    
    raw = cv2,imread(filePath,cv2.IMREAD_COLOR)
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


fail_list = []

imSizeX = 1280
imSizeY = 720
imCenter = 55+610/2

xBias = 280
yBias = 0

k = 0
for filePath in prnFile(PATH, PREFIX_BMP):
    k += 1
    raw = cv2.imread(filePath, cv2.IMREAD_COLOR) 
    # img = img_Contrast(img)
    imSize = 720
    img = raw[yBias:yBias+imSize, xBias:xBias+imSize]
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
    
    # Blur using 3 * 3 kernel. 
    gray_blurred = cv2.blur(gray, (3, 3)) 
    
    # Apply Hough transform on the blurred image. 
    detected_circles = cv2.HoughCircles(gray_blurred,  
                       cv2.HOUGH_GRADIENT, 1.5, 1270, param1 = 70, 
                   param2 = 40, minRadius = 315, maxRadius = 324) 
    # Draw circles that are detected. 
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
           

    
    # Draw circles that are detected. 
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
            
            #이미지가 저장될 때의 크기
            # img = cv2.resize(img, dsize=(sav_iSize, sav_iSize), interpolation=cv2.INTER_AREA)
            
            # cv2.imshow(f'hi', img)
            # cv2.waitKey(0)
            
            cv2.imwrite(PATH + 'cropped\\' + str(k) + PREFIX_PNG, img)
            # cv2.imwrite(PATH + 'cropped\\' + str(k) + PREFIX_PNG, img)
            print(f'{k} saved')
    else:
        fail_list.append(k)
        
print(f'failed: {fail_list}') 