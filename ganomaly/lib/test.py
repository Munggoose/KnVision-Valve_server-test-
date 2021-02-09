import matplotlib.pyplot as plt
import cv2
from PIL import Image
import numpy as np
from scipy.ndimage import rotate

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

def find_circle_crop(path:str):
    img = cv2.imread(path)
    height, width = img.shape[:2]
    mask = np.zeros((height, width))
    res = np.zeros((height, width, 3))
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # detect circles in the image
    circles = cv2.HoughCircles(gray, cv2.HOUGH_GRADIENT, 
                            1, 
                            100,
                            param1=50,param2=200,
                            minRadius=200, maxRadius=600)
    # ensure at least some circles were found
    if circles is not None:
        # convert the (x, y) coordinates and radius of the circles to integers
        circles = np.round(circles[0, :]).astype("int")
        # loop over the (x, y) coordinates and radius of the circles
        for (x, y, r) in circles:
            # draw the circle in the output image, then draw a rectangle
            # corresponding to the center of the circle
            cv2.circle(mask, (x, y), r, 255, cv2.FILLED)
            mask=mask.astype('uint8')
            _,thresh = cv2.threshold(mask,1,255,cv2.THRESH_BINARY)
            #Find Contour
            contours = cv2.findContours(thresh,cv2.RETR_EXTERNAL,cv2.CHAIN_APPROX_SIMPLE)
            #Find Reectangle
            x,y,w,h = cv2.boundingRect(np.array(contours[0][0]))
            #Crop masked_data
            crop = img[y:y+h,x:x+w]
            return make_rectangle(crop)

def seperate_image(img):
    print('seperate')
    res_list = find_circle_crop(img)
    
    for i in range(len(res_list)):
        res_list[i] = cv2.resize(res_list[i], (256, 256), interpolation=cv2.INTER_AREA)
    
    #res_list = np.array(res_list, np.dtype(object))
    print('seperate complete')
    return res_list

if __name__ == "__main__":
    #ax = [plt.subplot(2,2,i+1) for i in range(4)]
    res = find_circle_crop('./Untitled Folder/2.bmp')
    for idx, img in enumerate(res):
        size = 220
        plt.subplot(size+1+idx)
        plt.imshow(res[idx])
    plt.xticks([]), plt.yticks([])
    plt.show()
    
    print(res)