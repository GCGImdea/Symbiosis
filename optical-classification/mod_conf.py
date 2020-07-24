"""
Created on 2019-09-15
@author: Ricardo Padrino - github.com/rpadrino - IMDEA Networks
"""


class ClassifModels:

    # CONFIG.
    bin_model_img_width = 150
    bin_model_img_height = 150
    bin_model_weights_path = './weights/VGG16froz_lay256_x150_binary_fish_0.9809tstacc_20190507-1833.h5'

    multi_model_img_width = 190
    multi_model_img_height = 190
    multi_model_weights_path = './weights/multiclass_tinyNN_x190_adam_batchnorm_0.8653testacc_ep20_bs16_20190605-1504.h5'


    # VARIABLES
    #to load NN models and keep in memory
    binary_classification_model_vgg = None
    binary_classification_model_top = None
    binary_classification_model = None
    multi_classification_model = None
