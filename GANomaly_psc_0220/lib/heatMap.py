import cv2
import numpy as np
import argparse
from lib.Hough import img_Contrast

# def create_heatmap(im_map, im_cloud, kernel_size=(5,5),colormap=cv2.COLORMAP_JET,a1=0.5,a2=0.5):
def create_heatmap(im_map, im_cloud, colormap=cv2.COLORMAP_HOT, a1=0.5, a2=0.5):
    im_cloud_clr = cv2.applyColorMap(im_cloud, colormap)
    im_map = im_map + im_cloud_clr
    
    return im_map


def calc_diff(real_img, generated_img, batchsize, thres=1.0): # 0.7 for multi batch 0.67 for 0216
    """[summary]

    Args:
        real_img ([type]): [description]        shape = (3, 128, 128)
        generated_img ([type]): [description]   shape = (3, 128, 128)
        batchsize ([type]): [description]
        thres (float, optional): [description]  차영상의 한 픽셀의 차이가 thres보다 작을 때 0으로 만듦.

    Returns:
        [type]: [description]
    """
    
    diff_img = real_img - generated_img

    ch3_diff_img = diff_img
    
    # np.sum을 하여 R,G,B의 종합적인 차이를 구함.
    if batchsize == 1:
        diff_img = np.sum(diff_img, axis=0)
    else:
        diff_img = np.sum(diff_img, axis=1)
    diff_img = np.abs(diff_img)
    
    diff_img = np.log(diff_img + 1.5)
    if batchsize == 1:
        diff_img[diff_img < thres] = 0.0
    else:
        for bts in diff_img:
            bts[bts <= thres] = 0.0
        
    diff_img = cv2.normalize(diff_img, diff_img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
    
    return diff_img, ch3_diff_img


def Draw_Anomaly_image(real_img, diff_img, ch3_diff_img, batchsize):
    """[summary] calc_diff 로부터 구한 diff_img를 128x128에서 1280x720의 RAW Image에 적용.

    Args:
        real_img ([type]): [description]
        diff_img ([type]): [description]
        ch3_diff_img ([type]): [description]
        batchsize ([type]): [description]

    Returns:
        [type]: [description]
    """
    anomaly_img = real_img - ch3_diff_img 
    anomaly_img = cv2.normalize(anomaly_img, anomaly_img, 0, 255, cv2.NORM_MINMAX)
    
    diff_img = cv2.normalize(diff_img, diff_img, -2, 2, cv2.NORM_MINMAX).astype(np.uint8)
    diff_img = np.exp(diff_img/255)

    diff_img = cv2.normalize(diff_img, diff_img, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)

    diff_img_expanded = [diff_img, diff_img, diff_img]
    
    if batchsize == 1:
        anomaly_img = np.transpose(create_heatmap(np.transpose(anomaly_img, (1,2,0)), np.transpose(diff_img_expanded, (1,2,0)), a1=.2, a2=.8), (2,0,1))
    else:
        diff_img_expanded = np.transpose(diff_img_expanded, (1,0,3,2))
        for bts in range(batchsize):
            anomaly_img[bts] = np.transpose(create_heatmap(np.transpose(anomaly_img[bts], (1,2,0)), np.transpose(diff_img_expanded[bts], (1,2,0)), a1=.2, a2=.8), (2,1,0))
    
    return anomaly_img


def DrawResult(raw_img, diff_img, sav_fName, rawPATH):
        """[summary]: find_Center()를 활용해 raw Image의 중심점 찾고, 얻은 좌표 기반으로 diff_img 덧붙임.
        
        Related Functions:
            img_Contrast():raw_img의 이미지 대비 증가
            find_Center(): 입력받은 이미지에서 작은 원의 x,y좌표, r(반지름 반환)
        Args:
            raw_img ([type]): [description] shape=(720, 1280, 3)
            diff_img ([type]): [description] maybe shape=(3, w, h)
        """
        if sav_fName[0] == '0':
            rawPATH += 'normal\\' + sav_fName
        else:
            rawPATH += 'abnormal\\' + sav_fName

        raw_img = cv2.imread(rawPATH, cv2.IMREAD_COLOR)
        
        xBias = 510
        yBias = 220

        raw_img = img_Contrast(raw_img)
        imSize = 270
        img = raw_img[yBias:yBias+imSize, xBias:xBias+imSize]
        
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY) 
        gray_blurred = cv2.blur(gray, (3, 3)) 
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

                a = a + xBias
                b = b + yBias
                r += 204

                #검은 배경의 raw_diff 생성 후 위에서 얻은 a,b 좌표를 기준으로 하여 raw_img에 적용할 diff_img를 upsampling 하여 더함.
                raw_diff = np.zeros(shape=(1280,720))
                diff_img = cv2.resize(diff_img, dsize=(2*r, 2*r), interpolation=cv2.INTER_CUBIC)
                diff_img = np.transpose(diff_img, (1,0))
                raw_diff[a-r:a+r, b-r:b+r] = diff_img
                raw_diff = np.array([raw_diff, raw_diff, raw_diff])
                
                raw_img = cv2.normalize(raw_img, raw_img, 0, 255, cv2.NORM_MINMAX).get().astype(np.uint8)
                raw_diff = cv2.normalize(raw_diff, raw_diff, 0, 255, cv2.NORM_MINMAX).astype(np.uint8)
                
                raw_img = np.where(raw_diff == 0, raw_img, 0)
                
                raw_img = np.transpose(create_heatmap(np.transpose(raw_img, (1,2,0)), np.transpose(raw_diff, (1,2,0)), cv2.COLORMAP_HOT), (2,0,1))
                raw_img = raw_img[:, int(a-1.2*r):int(a+1.2*r), :]
                raw_img = np.transpose(raw_img, (2,1,0))
                
                return raw_img
        else:
            return None