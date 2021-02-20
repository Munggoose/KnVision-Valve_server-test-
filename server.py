import socket
# from ganomaly.lib.model import *  #model
# from ganomaly.models.ganomaly import Ganomaly
# from ganomaly.options import Options

import parameter
import torchvision.utils as vutils
import numpy as np
import os
import cv2
import json
import time
# from preprocessing import *

from os import walk
import torchvision.transforms as transforms


from GANomaly_psc.lib.Houghcopy import get_preprocess_img
from GANomaly_psc.lib.model import Ganomaly
from  GANomaly_psc.options import Options
from GANomaly_psc.lib.heatmap import DrawResult,calc_diff


host = parameter.addr
port = parameter.port


class server:
    def __init__(self, host, port):
        self.host = host
        self.port = port
        self.serv_addr = (self.host, self.port)
        self.conn_sock = None
        self.model = None
        self.dataloader = None
        self.sock = None
        self.modstr = None


    def establish(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.serv_addr)
        self.sock.listen(5)
        print('[server]server start')
        self.conn_sock, _ = self.sock.accept()

    def load_model(self, json_data):
        """[summary] 
        model initialize
        setting model pth 

        Args:
            json_data ([type]): [description] model setting parameter opt.parameter
        """
        if json_data == None:
            print('check')
            self.modstr = 'ganomaly'
            self.opt = Options().parse()
            self.opt.isize = 256
            self.opt.nz = 400
            self.opt.extralayers = 0
            
            self.model = Ganomaly(self.opt)
            self.model.load(parameter.weight_path)
            print('[server]default model ' + 'ganomaly' +' is ready')
        else:
            mod = json_data['mod']
            if mod == 'ganomaly':
                self.modstr = mod
                print('[server]select ' + mod)
                self.opt = Options().parse()
                self.opt.isize = json_data[mod]['isize']
                self.opt.nz = json_data[mod]['nz']
                self.opt.extralayers = json_data[mod]['extralayers']

                self.model = Ganomaly(self.opt,self.dataloader)
                self.model.load(parameter.weight_path)
                print('[server]model ' + mod +' is ready')
            else:
                print('[server]Can\'t find model ' + mod)

    def send_msg(self, msg):
        msg = msg.encode()
        self.conn_sock.sendall(msg)

    def recv_msg(self, size = 1024):
        msg = self.conn_sock.recv(size)
        if not msg:
            self.conn_sock.close()
            exit()
        return msg.decode()

    
    def test_img(self, input_img):
        """[summary] return Ganomaly abnormal score

        Args:
            input_img ([type]):preprocessed image data

        Returns:
            [float]: abnormal score, 
            [numpy]: fake_image
        """
        if self.modstr == 'ganomaly':
            # norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))   
            img = cv2.resize(input_img, dsize=(128, 128)) #opt.isize
            #img_t = transforms.CenterCrop(128)(img_t)
            img_t = transforms.ToTensor()(img)
            # img_t = norm(img_t)

            err, fake = self.model.test_one(img_t)
            # np_fake = fake.cpu().numpy()
            np_fake = fake
            return err.item(), np_fake
        else:
            print('[server]model isn\'t ready')


    def disconnet(self):
        self.conn_sock.close()
        self.sock.close()


if __name__ == '__main__':
    try:
    with open(parameter.json_path, 'r') as f:
        if f is None:
            print('[server]cannot find json file in ' + parameter.json_path)
            json_data = None
        json_data = json.load(f)
    except:
        print('[server]cannot find json file in ' + parameter.json_path)
        mod = input('[server]select model <now available - ganomaly> :')
    

    ## server and model init
    S = server(host, port)
    S.load_model(json_data)
    S.establish()

    S.send_msg('server is ready')   #send confirm msg

    cnt = 0

    while True:
        err_scores = []
        str_err_scores = ""
        img_path = S.recv_msg()
        print('Connect')
        print('[server]img path(or quit) is : ' + img_path)
        if not img_path:
            # print('No path')
            assert('NO Path')

        if img_path == 'finish':
            S.disconnet()
            break
        
        start = time.process_time()

        target_img, _check = get_preprocess_img(img_path)
        # print('data', np.shape(target_img))
        # print(len(target_img))
        if not _check:
            str_err_scores += f"Can't find circle "
            diagnosis_result = 'Abnormal'
            str_err_scores += f"  result: {diagnosis_result}"
            # S.send_msg(str_err_scores)
            S.send_msg(diagnosis_result)
            end = time.process_time()
            #print('Not found Circle ')
            print('Abnormal')
            print('[server]processing time : ', (end - start))
            continue
        
        
        
        thresholds = 4 #0.05
        diagnosis_result = 'Normal'

        err, fake_img = S.test_img(target_img)
        diff_img = calc_diff(target_img, fake_img, 1, thres=0.67)
        result_img = DrawResult(diff_img, img_path)
        cv2,imwrite('./sample.bmp',result_img)
        # cv2.imwrite('./sample.bmp',target_img)
        
        
        if err > thresholds:
            diagnosis_result = 'Abnormal'
        str_err_scores += "{:.2f} ".format(err)
        str_err_scores += f"  result: {diagnosis_result}"
        # S.send_msg(str_err_scores)
        S.send_msg(diagnosis_result)
        end = time.process_time()
        print('err_scores: ', err ,' rsult: ',diagnosis_result)
        print('[server]processing time : ', (end - start))

    S.disconnet()
        
