import cv2 
import numpy as np 
import os
import os.path
import torch
import torchvision.transforms as transforms
"""
        인풋이미지는 큰비젼 카메라로 찍은 것.

    Returns:
        sav_iSize 크기의 1구간 사진.
        1구간을 제외한 다른 부분은 검은색 마스킹 처리리
"""

class HoughCircleDetection():
    def __init__(self, PREFIX, isize):
        # self.PATH = PATH
        self.PREFIX_BMP = '.bmp'
        self.PREFIX_PNG = '.png'
        self.PREFIX = PREFIX
        self.sav_iSize = 128
        pass
    
    def prnFile(rootDir, prefix):
        files = os.listdir(rootDir)
        tmp = []
        for file in files:
            path = os.path.join(rootDir, file)
            if(file[-4:] == prefix):
                tmp.append(path)
        return tmp
    
    def img_Contrast(self, img):
        # raw = np.zeros(shape=(3,1280,720))
        # raw = img[0].numpy()
        # raw = cv2.normalize(raw, raw, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        # raw = np.transpose(raw, (1,2,0))
        
        lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
        # lab = cv2.cvtColor(img, cv2.COLOR_)
        # lab = img
        l, a, b = cv2.split(lab)
        clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
        cl = clahe.apply(l)

        limg = cv2.merge((cl, a, b))
        final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
        return final
    
    
    
    
    def find_Center(self, raw):
        fail_list = []

        imSizeX = 1280
        imSizeY = 720
        imCenter = 55+610/2

        xBias = 510
        yBias = 220
        
        raw = self.img_Contrast(raw)
        # raw = cv2.normalize(raw, raw, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imSize = 270
        img = raw[yBias:yBias+imSize, xBias:xBias+imSize]
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 

        # Blur using 3 * 3 kernel. 
        gray_blurred = cv2.blur(gray, (3, 3)) 

        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1.5, 1270, param1 = 95, 
                       param2 = 30, minRadius = 98, maxRadius = 102) 

        # Draw circles that are detected. 
        if detected_circles is not None: 
         
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                # print(r)


                cv2.circle(img, (a, b), r, (255,0,255), 1)

                a += xBias
                b += yBias
                r_small = r
                r += 204



                backGD_circle = np.zeros((imSizeY,imSizeX, 3), dtype="uint8")
                cv2.circle(backGD_circle, (a, b), r, (255,255,255), -1)
                rectX = (a - r) 
                rectY = (b - r)
                raw = cv2.bitwise_and(backGD_circle, raw)


                backGD_circle_small = np.full((imSizeY, imSizeX, 3), 255, dtype='uint8')
                cv2.circle(backGD_circle_small, (a, b), r_small, (0,0,0), -1)

                raw = cv2.bitwise_and(backGD_circle_small, raw)
                img = raw[rectY:(rectY+2*r), rectX:(rectX+2*r)]

                #이미지가 저장될 때의 크기
                img = cv2.resize(img, dsize=(self.sav_iSize, self.sav_iSize), interpolation=cv2.INTER_AREA)
                img = np.transpose(img, (2,0,1))
                img = cv2.normalize(img, img, 0, 1, cv2.NORM_MINMAX).get().astype(np.uint8)
                # img = (img - 0.5)  / 0.5
                # transform = transforms.Compose([
                #     transforms.Normalize(
                #         mean=[0.5, 0.5, 0.5],
                #         std=[0.5, 0.5, 0.5],
                #     ),
                # ])
                # img = transform(torch.Tensor([img]))
        else:
            print(f'failed: {fail_list}') 
        return torch.Tensor([img])
    
    def DrawResult(self, raw_img, diff_img, ch3_diff):
        """[summary]
        Args:
            raw_img ([type]): [description] shape=(720, 1280, 3)
            diff_img ([type]): [description] maybe shape=(3, w, h)
            ch3_diff ([type]): [description] maybe shape=(3, w, h)
        """
        xBias = 510
        yBias = 220

        raw_img = self.img_Contrast(raw_img)
        # raw = cv2.normalize(raw, raw, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
        imSize = 270
        img = raw_img[yBias:yBias+imSize, xBias:xBias+imSize]
        cv2.imshow('raw', img)
        cv2.waitKey(0)
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        # Blur using 3 * 3 kernel. 
        gray_blurred = cv2.blur(gray, (3, 3)) 
        # Apply Hough transform on the blurred image. 
        detected_circles = cv2.HoughCircles(gray_blurred,  
                           cv2.HOUGH_GRADIENT, 1.5, 1270, param1 = 95, 
                       param2 = 30, minRadius = 98, maxRadius = 102) 
        # Draw circles that are detected. 
        if detected_circles is not None: 
            raw_img = np.transpose(raw_img, (2,1,0)) # -> (3, 1280, 720)
            # Convert the circle parameters a, b and r to integers. 
            detected_circles = np.uint16(np.around(detected_circles)) 
            for pt in detected_circles[0, :]: 
                a, b, r = pt[0], pt[1], pt[2] 
                print(r)

                a = a + xBias
                b = b + yBias
                r += 204

                raw_diff = np.zeros(shape=(1280,720))
                print(np.shape(raw_diff))
                print(np.shape(raw_diff[a:a+2*r, b:b+2*r]))
                raw_diff[a-r:a+r, b-r:b+r] = np.resize(diff_img, (2*r, 2*r))
                # raw_diff[a:a+2*r, b:b+2*r] = np.resize(diff_img, (2*r, 2*r))
                raw_diff = [raw_diff, raw_diff, raw_diff]
                
                cv2.imshow('raw_diff', np.transpose(raw_diff,(1,2,0))) # [H W C]
                cv2.waitKey(0)
                
                ch3_diff = np.resize(ch3_diff, (3, 2*r, 2*r))
                
                ch3_diff = cv2.normalize(ch3_diff, ch3_diff, 0, 255, cv2.NORM_MINMAX)
                
                print(f'raw: {np.average(raw_img)}')
                print(f'raw: {np.min(raw_img)}')
                print(f'raw: {np.max(raw_img)}')
                print()
                
                

                raw_img[:,a:a+2*r, b:b+2*r] = raw_img[:,a:a+2*r, b:b+2*r] - ch3_diff
                raw_img = np.transpose(raw_img + raw_diff, (2,1,0))
                
                raw_img = cv2.normalize(raw_img, raw_img, 0, 255, cv2.NORM_MINMAX).get().astype(np.uint8)
                
        return raw_img
    
def img_Contrast(img):
        
    lab = cv2.cvtColor(img, cv2.COLOR_BGR2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=3, tileGridSize=(8,8))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    final = cv2.cvtColor(limg, cv2.COLOR_LAB2BGR)
    return final