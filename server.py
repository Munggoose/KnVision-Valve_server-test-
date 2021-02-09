import socket
# from ganomaly.lib.model import *  #model
from ganomaly.models.ganomaly import Ganomaly
from ganomaly.options import Options

import parameter
import torchvision.utils as vutils
import numpy as np
import os
import cv2
import json
import time
from preprocessing import *
from os import walk
import torchvision.transforms as transforms

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
        # self.transforms = transforms.Compose(
        #     [
        #         transforms.Resize(256),
        #         transforms.ToTensor(),
        #         transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
        #     ]
        # )

    def establish(self):
        self.sock = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
        self.sock.bind(self.serv_addr)
        self.sock.listen(5)
        print('[server]server start')
        self.conn_sock, _ = self.sock.accept()

    def load_model(self, json_data):
        if json_data == None:
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
        if self.modstr == 'ganomaly':
            norm = transforms.Normalize((0.5,0.5,0.5),(0.5,0.5,0.5))   
            img = cv2.resize(input_img, dsize=(256, 256))
            img_t = transforms.ToTensor()(img)
            img_t = norm(img_t)

            # img_t = self.transforms(input_img)

            err, fake, _ = self.model.test_one(img_t)
            np_fake = fake.cpu().numpy()
            return err.item(), np_fake
        else:
            print('[server]model isn\'t ready')


    def disconnet(self):
        self.conn_sock.close()
        self.sock.close()


host = parameter.addr
port = parameter.port

try:
    with open(parameter.json_path, 'r') as f:
        if f is None:
            print('[server]cannot find json file in ' + parameter.json_path)
            json_data = None
        json_data = json.load(f)
except:
    print('[server]cannot find json file in ' + parameter.json_path)
    mod = input('[server]select model <now available - ganomaly> :')
    

S = server(host, port)
S.load_model(json_data)
S.establish()

S.send_msg('server is ready')   #send confirm msg

cnt = 0

while True:
    err_scores = []
    str_err_scores = ""
    img_path = S.recv_msg()
    print('check')
    print('[server]img path(or quit) is : ' + img_path)
    if not img_path:
        # print('No path')
        assert('NO Path')

    if img_path == 'finish':
        S.disconnet()
        break
    
    start = time.process_time()
    # img = cv2.imread(img_path)
    # target_img = seperate_image(img)  ##preprocessing
    target_img = get_preprocess_img(img_path)
    cv2.imwrite('./sample.bmp',target_img)
    
    thresholds = 0.021
    diagnosis_result = 'Normal'

    # err, _ = S.test_img(img)
    # if err > thresholds:
    #     diagnosis_result = 'Abnormal'
    # str_err_scores += "{:.2f} ".format(err)

    # for index, sep_img in enumerate(sep_imgs):
    err, _ = S.test_img(target_img)
    if err > thresholds:
        diagnosis_result = 'Abnormal'
    str_err_scores += "{:.2f} ".format(err)
    str_err_scores += f"  result: {diagnosis_result}"
    S.send_msg(str_err_scores)
    end = time.process_time()
    print('[server]processing time : ', (end - start))

S.disconnet()