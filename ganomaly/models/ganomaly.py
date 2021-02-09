from collections import OrderedDict
import os
import time
import numpy as np
from tqdm import tqdm
import random

from torch.autograd import Variable
import torch.optim as optim
import torch.nn as nn
import torch.utils.data
import torchvision.utils as vutils
#Custom lib
from ganomaly.lib.networks import NetG, NetD, weights_init
from ganomaly.lib.visualizer import Visualizer
from ganomaly.lib.loss import l2_loss
from ganomaly.lib.evaluate import evaluate
# from ganomaly.lib.visualizer import compare_images

class BaseModel():
    """BaseModel for ganomaly
    """
    def __init__(self,opt,dataloader):

        #Seed for deteministic behavior
        self.seed(opt.manualseed)
        self.visualizer = Visualizer(opt)
        #Initialize variables
        self.opt = opt
        self.dataloader = dataloader
        self.device = torch.device("cuda:0" if self.opt.device != 'cpu' else "cpu")

    def set_input(self, input:torch.Tensor):
        """Set input and ground truth

        Args:
            input (torch.Tensor): Input data for batch i
        """
        with torch.no_grad():
            self.input.resize_(input[0].size()).copy_(input[0])
            self.gt.resize_(input[1].size()).copy_(input[1])

    def seed(self, seed_value):
        """ Seed setting

        Args:
            seed_value (int): [description]
        """
        if seed_value == -1:
            return
        

        #Otherwise seed all functionality
        random.seed(seed_value)
        torch.manual_seed(seed_value)
        torch.cuda.manual_seed_all(seed_value)
        np.random.seed(seed_value)
        torch.backends.cudnn.deteministic = True

    def get_errors(self):
        """Get netD and netG errors.

        Returns:
        [OrdertDict]: Dictionary containing errors
        """
        
        errors = OrderedDict([
            ('err_d', self.err_d.item()),
            ('err_g', self.err_g.item()),
            ('err_g_adv', self.err_g_adv.item()),
            ('err_g_con', self.err_g_con.item()),
            ('err_g_enc', self.err_g_enc.item())])

        return errors

    # def compare_fake_real(self, score):
    #     reals,fakes,_ = self.get_current_images()
    #     for (real,fake) in (reals,fakes):
    #         compare_images(real_img=real, generated_img=fake, score=score)

    def get_current_images(self):
        """return current images.

        Returns:
            [real,fake,fixed]
        """
        reals = self.input.data
        fakes = self.fake.data
        fixed = self.netg(self.fixed_input)[0].data

        return reals,fakes, fixed
    
    def save_weights(self,epoch):
        """Save NetG and netD weights for the current epoch

        Args:
            epoch ([int]): Current epoch number
        """
        
        weight_dir = os.path.join(self.opt.outf, self.opt.name, 'train', 'weights')
        if not os.path.exists(weight_dir): os.makedirs(weight_dir)

        torch.save({'epoch': epoch + 1, 'state_dict': self.netg.state_dict()},
                f'{weight_dir}/acc_{self.best_auc} epoch{epoch+1} netG err_g {self.err_g}.pth')
        torch.save({'epoch': epoch + 1, 'state_dict': self.netd.state_dict()},
                f'{weight_dir}/acc_{self.best_auc} epoch{epoch+1} netD err_d {self.err_d}.pth')
    
    def train_one_epoch(self):
        """ Train the model for one epoch.
        """
        self.netg.train()
        epoch_iter = 0
        for data in tqdm(self.dataloader['train'],leave =False, total=len(self.dataloader['train'])):
            self.total_steps += self.opt.batchsize
            epoch_iter += self.opt.batchsize

            self.set_input(data)
            #self.optimize()
            self.optimize_params()  ## forward()
            
            if self.total_steps % self.opt.print_freq == 0:
                errors = self.get_errors()
                if self.opt.display:
                    counter_ratio = float(epoch_iter) / len(self.dataloader['train'].dataset)
                    self.visualizer.plot_current_errors(self.epoch, counter_ratio, errors)

            if self.total_steps % self.opt.save_image_freq == 0:
                reals, fakes, fixed = self.get_current_images()
                # self.visualizer.save_current_images(self.epoch, reals, fakes, fixed)
                if self.opt.display:
                    self.visualizer.display_current_images(reals, fakes, fixed)
                    
        print(f">> D_loss: {self.err_d} , G_loss: {self.err_g}")
        print(">> Training model %s. Epoch %d/%d" % (self.name, self.epoch+1, self.opt.niter))
        
    
    def train(self):
        """Train model
        """
        self.total_step = 0
        self.best_auc = 0

        #Train for niter epochs
        print(">> Training model %s." % self.name)
        for self.epoch in range(self.opt.iter, self.opt.niter):
            # Train for one epoch
            self.train_one_epoch()

            res = self.test()
            # if self.epoch > self.opt.save_limit_epoch:
            # if res[self.opt.metric] > best_auc:
            if (res[self.opt.metric] > self.best_auc) or (self.epoch % 50 == 0):
                self.best_auc = res[self.opt.metric]
                print("best_auc: %6f" % self.best_auc)
                self.save_weights(self.epoch)
        print(">> Training model %s.[Done]" % self.name)

    def test(self):
        """ Test GANomaly model.
        Args:
            dataloader ([type]): Dataloader for the test set
        Raises:
            IOError: Model weights not found.
        """
        with torch.no_grad():
            # Load the weights of netg and netd.
            if self.opt.load_weights:
                path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
                pretrained_dict = torch.load(path)['state_dict']

                try:
                    self.netg.load_state_dict(pretrained_dict)
                except IOError:
                    raise IOError("netG weights not found")
                print('   Loaded weights.')

            self.opt.phase = 'test'

            # Create big error tensor for the test set.
            self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
            self.gt_labels = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.long,    device=self.device)
            self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
            self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

            # print("   Testing model %s." % self.name)
            self.times = []
            self.total_steps = 0
            epoch_iter = 0
            for i, data in enumerate(self.dataloader['test'], 0):
                self.total_steps += self.opt.batchsize
                epoch_iter += self.opt.batchsize
                time_i = time.time()
                self.set_input(data)
                self.fake, latent_i, latent_o = self.netg(self.input)

                error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
                time_o = time.time()

                self.an_scores[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = error.reshape(error.size(0))
                self.gt_labels[i*self.opt.batchsize : i*self.opt.batchsize+error.size(0)] = self.gt.reshape(error.size(0))
                self.latent_i [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_i.reshape(error.size(0), self.opt.nz)
                self.latent_o [i*self.opt.batchsize : i*self.opt.batchsize+error.size(0), :] = latent_o.reshape(error.size(0), self.opt.nz)

                self.times.append(time_o - time_i)

                # Save test images.
                if self.opt.save_test_images:
                    dst = os.path.join(self.opt.outf, self.opt.name, 'test', 'images')
                    if not os.path.isdir(dst):
                        os.makedirs(dst)
                    real, fake, _ = self.get_current_images()
                    vutils.save_image(real, '%s/real_%03d.eps' % (dst, i+1), normalize=True)
                    vutils.save_image(fake, '%s/fake_%03d.eps' % (dst, i+1), normalize=True)

            # Measure inference time.
            self.times = np.array(self.times)
            self.times = np.mean(self.times[:100] * 1000)

            # Scale error vector between [0, 1]
            self.an_scores = (self.an_scores - torch.min(self.an_scores)) / (torch.max(self.an_scores) - torch.min(self.an_scores))
            # auc, eer = roc(self.gt_labels, self.an_scores)
            auc = evaluate(self.gt_labels, self.an_scores, metric=self.opt.metric)
            performance = OrderedDict([('Avg Run Time (ms/batch)', self.times), (self.opt.metric, auc)])
            # print(auc)

            if self.opt.display_id > 0 and self.opt.phase == 'test':
                counter_ratio = float(epoch_iter) / len(self.dataloader['test'].dataset)
                self.visualizer.plot_performance(self.epoch, counter_ratio, performance)
            return performance

    def predict(self,data):
        """predict data's abnomal score

        Args:
            data ([type]): test data

        Returns:
            [type]: abnomal score
        """
        # pretrained_dict = torch.load(w_path)['state_dict'] 
        self.netg.eval()
        self.opt.phase = 'test'

        # Create big error tensor for the test set.
        self.an_scores = torch.zeros(size=(len(self.dataloader['test'].dataset),), dtype=torch.float32, device=self.device)
        self.latent_i  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)
        self.latent_o  = torch.zeros(size=(len(self.dataloader['test'].dataset), self.opt.nz), dtype=torch.float32, device=self.device)

        # print("   Testing model %s." % self.name)
        self.set_input(data)
        self.fake, latent_i, latent_o = self.netg(self.input)
        error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)
        
        return error

    def set_weight(self,path_g, path_d):
        """Set trained model weight 
        """
        
        # path = "./output/{}/{}/train/weights/netG.pth".format(self.name.lower(), self.opt.dataset)
        path = path_g
        pretrained_dict_g = torch.load(path)['state_dict']
        # path = "./output/{}/{}/train/weights/netD.pth".format(self.name.lower(), self.opt.dataset)
        path = path_d
        pretrained_dict_d = torch.load(path)['state_dict']
        try:
            self.netg.load_state_dict(pretrained_dict_g)
            self.netd.load_state_dict(pretrained_dict_d)
        except IOError:
            raise IOError("weights not found")
        print(' Loaded weights.')
    
    def load(self, weight_path:str):
        """load and set model weight 
        Args:
            weight_path ([str]): path .pth 
        Raises:
            IOError: [description]
        """
        with torch.no_grad():
            path = weight_path + '\\netG.pth'
            #print(path)
            pretrained_dict = torch.load(path)['state_dict']
            try:
                self.netg.load_state_dict(pretrained_dict)
            except IOError:
                raise IOError("[server]netG weights not found")
            print('[server]   Loaded weights.')
            self.opt.phase = 'test'

    def test_one(self, img_):
        """predict abnomlay score from image

        Args:
            img_ (Image): [description]

        Returns:
            [type]: [description]
        """
        with torch.no_grad():

            self.input.resize_(img_.size()).copy_(img_).unsqueeze_(0)

            self.fake, latent_i, latent_o = self.netg(self.input)

            self.diff = None
            '''test : image diff
            self.diff = torch.abs(self.input - self.fake)
            vutils.save_image(self.input, 'C:\\Users\\HCI\\Documents\\noahs_ark\\YSH\\lab\\datas\\client_server2\\test1\\read' + str(cnt) +'.jpg' ,normalize=True)
            vutils.save_image(self.diff, 'C:\\Users\\HCI\\Documents\\noahs_ark\\YSH\\lab\\datas\\client_server2\\test1\\diff' + str(cnt) +'.jpg' ,normalize=True)
            self.diff = self.diff.cpu().numpy()
            '''

            error = torch.mean(torch.pow((latent_i-latent_o), 2), dim=1)  
            '''
            vutils.save_image(self.fake, 'C:\\Users\\HCI\\Documents\\noahs_ark\\YSH\\lab\\datas\\client_server2\\test1\\fake' + str(cnt) +'.jpg' ,normalize=True)
            '''
        return error, self.fake, self.diff



class Ganomaly(BaseModel):
    """GANomaly Class
    """

    @property
    def name(self): return 'Ganomaly'

    def __init__(self, opt, dataloader):
        super(Ganomaly, self).__init__(opt, dataloader)

        # -- Misc attributes
        self.epoch = 0
        self.times = []
        self.total_steps = 0

        ##
        # Create and initialize networks.
        self.netg = NetG(self.opt).to(self.device)
        self.netd = NetD(self.opt).to(self.device)
        self.netg.apply(weights_init)
        self.netd.apply(weights_init)

        ##
        if self.opt.resume != '':
            print("\nLoading pre-trained networks.")
            self.opt.iter = torch.load(os.path.join(self.opt.resume, 'netG.pth'))['epoch']
            self.netg.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netG.pth'))['state_dict'])
            self.netd.load_state_dict(torch.load(os.path.join(self.opt.resume, 'netD.pth'))['state_dict'])
            print("\tDone.\n")

        #set loss_func
        self.l_adv = l2_loss
        self.l_con = nn.L1Loss()
        self.l_enc = l2_loss
        self.l_bce = nn.BCELoss()

        ##
        # Initialize input tensors.
        self.input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.label = torch.empty(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.gt    = torch.empty(size=(opt.batchsize,), dtype=torch.long, device=self.device)
        self.fixed_input = torch.empty(size=(self.opt.batchsize, 3, self.opt.isize, self.opt.isize), dtype=torch.float32, device=self.device)
        self.real_label = torch.ones (size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        self.fake_label = torch.zeros(size=(self.opt.batchsize,), dtype=torch.float32, device=self.device)
        ##
        # Setup optimizer
        if self.opt.isTrain:
            self.netg.train()
            self.netd.train()
            self.optimizer_d = optim.Adam(self.netd.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))
            self.optimizer_g = optim.Adam(self.netg.parameters(), lr=self.opt.lr, betas=(self.opt.beta1, 0.999))

    ##
    def forward_g(self):
        """ Forward propagate through netG
        self.latent_i -> ouput encoder1 latent_z
        self.latent_o -> ouput encoder2
        self.fake -> genarator image
        
        """
        self.fake, self.latent_i, self.latent_o = self.netg(self.input)

    ##
    def forward_d(self):
        """ Forward propagate through netD
        self.input -> real_img
        pred_ -> predict score
        feat_ -> features
        """
        self.pred_real, self.feat_real = self.netd(self.input)
        self.pred_fake, self.feat_fake = self.netd(self.fake.detach())

    ##
    def backward_g(self):
        """ Backpropagate through netG
        err_ : loss value
        w_ : loss weight
        """
        self.err_g_adv = self.l_adv(self.netd(self.input)[1], self.netd(self.fake)[1]) #loss by discriminator
        self.err_g_con = self.l_con(self.fake, self.input) #loss by image
        self.err_g_enc = self.l_enc(self.latent_o, self.latent_i)  # loss by latent_z
        self.err_g = self.err_g_adv * self.opt.w_adv + \
                     self.err_g_con * self.opt.w_con + \
                     self.err_g_enc * self.opt.w_enc
        self.err_g.backward(retain_graph=True)

    ##
    def backward_d(self):
        """ Backpropagate through netD
        """
        # Real - Fake Loss
        self.err_d_real = self.l_bce(self.pred_real, self.real_label)
        self.err_d_fake = self.l_bce(self.pred_fake, self.fake_label)

        # NetD Loss & Backward-Pass
        self.err_d = (self.err_d_real + self.err_d_fake) * 0.5
        self.err_d.backward()

    ##
    def reinit_d(self):
        """ Re-initialize the weights of netD
        """
        self.netd.apply(weights_init)
        #print('  Reloading net d')

    def optimize_params(self):
        """ Forwardpass, Loss Computation and Backwardpass.f
        """
        
        self.optimizer_g.zero_grad()
        self.optimizer_d.zero_grad()
        
        # Forward-pass
        self.forward_g()
        self.forward_d()

        # Backward-pass
        # netg
        self.backward_g()
        self.optimizer_g.step()

        # netd
        self.backward_d()
        self.optimizer_d.step()

        if self.err_d.item() < 1e-5:
            self.reinit_d()