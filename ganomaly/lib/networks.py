""" Network architectures.
"""
# pylint: disable=W0221,W0622,C0103,R0913

##
import torch
import torch.nn as nn
import torch.nn.parallel

##
def weights_init(mod):
    """
    Custom weights initialization called on netG, netD and netE
    :param mod: model module
    :return:
    """
    classname = mod.__class__.__name__
    if classname.find('Conv') != -1:
        mod.weight.data.normal_(0.0, 0.02)
    elif classname.find('BatchNorm') != -1:
        mod.weight.data.normal_(1.0, 0.02)
        mod.bias.data.fill_(0)

###
class Encoder(nn.Module):
    """
    DCGAN ENCODER NETWORK
    """

    def __init__(self, img_size, n_latent_dim, n_channel, ndf, ngpu, n_extra_layers=2, add_final_conv=True):
        super(Encoder, self).__init__()
        self.ngpu = ngpu
        assert img_size % 16 == 0, "isize has to be a multiple of 16"

        main = nn.Sequential()
        # input is n_channel x img_size x img_size
        main.add_module('initial-conv-{0}-{1}'.format(n_channel, ndf),
                        nn.Conv2d(n_channel, ndf, 4, 2, 1, bias=False))
        main.add_module('initial-relu-{0}'.format(ndf),
                        nn.LeakyReLU(0.2, inplace=True))
        cur_size, cur_ndf = img_size / 2, ndf
        
        # feature extractor
        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cur_ndf),
                            nn.Conv2d(cur_ndf, cur_ndf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cur_ndf),
                            nn.BatchNorm2d(cur_ndf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cur_ndf),
                            nn.LeakyReLU(0.2, inplace=True))

        while cur_size > 4:
            in_feat = cur_ndf
            out_feat = cur_ndf * 2
            main.add_module('pyramid-{0}-{1}-conv'.format(in_feat, out_feat),
                            nn.Conv2d(in_feat, out_feat, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(out_feat),
                            nn.BatchNorm2d(out_feat))
            main.add_module('pyramid-{0}-relu'.format(out_feat),
                            nn.LeakyReLU(0.2, inplace=True))
            cur_ndf = cur_ndf * 2
            cur_size = cur_size / 2

        # state size. K x 4 x 4(?)
        if add_final_conv:
            main.add_module('final-{0}-{1}-conv'.format(cur_ndf, 1),
                            nn.Conv2d(cur_ndf, n_latent_dim, 4, 1, 0, bias=False))

        self.main = main

    def forward(self, input):
        
        if self.ngpu > 1: #if use multiple gpu
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)

        return output

##
class Decoder(nn.Module):
    """
    DCGAN DECODER NETWORK
    """
    def __init__(self, img_size, n_latent_dim, n_channel, ngf, ngpu, n_extra_layers=0, ti_size= 4):
        super(Decoder, self).__init__()
        self.ngpu = ngpu
        assert img_size % 16 == 0, "isize has to be a multiple of 16"
        #tisize: Encoder''s output size
        cur_ngf, tisize = ngf // 2, ti_size
        while tisize != img_size:
            cur_ngf = cur_ngf * 2
            tisize = tisize * 2

        main = nn.Sequential()
        # input is Z, going into a convolution
        main.add_module('initial-{0}-{1}-convt'.format(n_latent_dim, cur_ngf),
                        nn.ConvTranspose2d(n_latent_dim, cur_ngf, 4, 1, 0, bias=False))
        main.add_module('initial-{0}-batchnorm'.format(cur_ngf),
                        nn.BatchNorm2d(cur_ngf))
        main.add_module('initial-{0}-relu'.format(cur_ngf),
                        nn.ReLU(True))

        cur_size, _ = 4, cur_ngf
        while cur_size < img_size // 2:
            main.add_module('pyramid-{0}-{1}-convt'.format(cur_ngf, cur_ngf // 2),
                            nn.ConvTranspose2d(cur_ngf, cur_ngf // 2, 4, 2, 1, bias=False))
            main.add_module('pyramid-{0}-batchnorm'.format(cur_ngf // 2),
                            nn.BatchNorm2d(cur_ngf // 2))
            main.add_module('pyramid-{0}-relu'.format(cur_ngf // 2),
                            nn.ReLU(True))
            cur_ngf = cur_ngf // 2
            cur_size = cur_size * 2

        # Extra layers
        for t in range(n_extra_layers):
            main.add_module('extra-layers-{0}-{1}-conv'.format(t, cur_ngf),
                            nn.Conv2d(cur_ngf, cur_ngf, 3, 1, 1, bias=False))
            main.add_module('extra-layers-{0}-{1}-batchnorm'.format(t, cur_ngf),
                            nn.BatchNorm2d(cur_ngf))
            main.add_module('extra-layers-{0}-{1}-relu'.format(t, cur_ngf),
                            nn.ReLU(True))

        main.add_module('final-{0}-{1}-convt'.format(cur_ngf, n_channel),
                        nn.ConvTranspose2d(cur_ngf, n_channel, 4, 2, 1, bias=False))
        main.add_module('final-{0}-tanh'.format(n_channel),
                        nn.Tanh())
        self.main = main

    def forward(self, input):
        if self.ngpu > 1:
            output = nn.parallel.data_parallel(self.main, input, range(self.ngpu))
        else:
            output = self.main(input)
            
        return output


##
class NetD(nn.Module):
    """
    DISCRIMINATOR NETWORK
    """

    def __init__(self, opt):
        super(NetD, self).__init__()
        model = Encoder(opt.isize, 1, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        layers = list(model.main.children())

        self.features = nn.Sequential(*layers[:-1])
        self.classifier = nn.Sequential(layers[-1])
        self.classifier.add_module('Sigmoid', nn.Sigmoid())

    def forward(self, x):
        features = self.features(x)
        features = features
        classifier = self.classifier(features)
        classifier = classifier.view(-1, 1).squeeze(1)

        return classifier, features


##
class NetG(nn.Module):
    """
    GENERATOR NETWORK
    """

    def __init__(self, opt):
        super(NetG, self).__init__()
        self.encoder1 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.decoder = Decoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)
        self.encoder2 = Encoder(opt.isize, opt.nz, opt.nc, opt.ngf, opt.ngpu, opt.extralayers)

    def forward(self, x):
        latent_i = self.encoder1(x)
        gen_imag = self.decoder(latent_i)
        latent_o = self.encoder2(gen_imag)
        return gen_imag, latent_i, latent_o