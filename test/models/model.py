# modified from https://github.com/tensorflow/models/blob/master/research/slim/nets/s3dg.py
#from sklearn.metrics import log_loss
import json
import torch.nn as nn
import torch
from .transformer import *
from einops import rearrange

'''
args.N
args.d_model
args.res_base_model
args.H 
args.num_queries
args.dropout
args.attribute_set_size
'''


class MRIBrain(nn.Module):

    def __init__(self, config, mode='train'):
        super(MRIBrain, self).__init__()

        self.mode = mode
        self.d_model = config['d_model']
        
        self.cl_fc = nn.Linear(config['out_feature'],768)
        self.excluded_disease = ['normal']
        self.disease_name = json.load(open(config['disease_order'],'r'))
        if "exclude_class" in config and config["exclude_class"]:
            self.excluded_disease += config["exclude_classes"]
            self.keep_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in config["exclude_classes"] ]   
        self.cl_class_dim = [self.disease_name.index(i) for i in self.disease_name if i not in self.excluded_disease ]   
        
        
        ''' visual backbone'''
        # self.resnet_dict = {"resnet18": models.resnet18(pretrained=False),
        #                     "resnet50": models.resnet50(pretrained=False)}
        # resnet = self._get_res_basemodel(config['res_base_model'])
        # num_ftrs = int(resnet.fc./2)
        # self.res_features = nn.Sequential(*list(resnet.children())[:-3])

        ###################################
        ''' Query Decoder'''
        ###################################

        self.H = config['H'] 
        decoder_layer = TransformerDecoderLayer(self.d_model, config['H'] , 1024,
                                        0.1, 'relu',normalize_before=True)
        self.decoder_norm = nn.LayerNorm(self.d_model)
        self.decoder = TransformerDecoder(decoder_layer, config['N'] , self.decoder_norm,
                                  return_intermediate=False)

        # Learnable Queries
        #self.query_embed = nn.Embedding(config['num_queries'] ,self.d_model)
        self.dropout_feas = nn.Dropout(config['dropout'] )

        # Attribute classifier
        self.classifier = nn.Parameter(torch.empty(config['num_classes'],self.d_model,config['attribute_set_size']))
        torch.nn.init.normal_(self.classifier, mean=0.0, std=0.02)

        self.apply(self._init_weights)

        # focal_loss_weight
        # self.class_weight = json.load(open(config['class_weight'],'r'))
        # LA
        # disease_order=json.load(open(config['disease_order'],'r'))
        # class_p = json.load(open(config['class_p'],'r'))
        # self.class_p = torch.tensor(config["la_alpha"])*torch.log(torch.tensor([[class_p[i][0]/class_p[i][1]] for i in disease_order]))
        self.config = config
    
    def forward(self, image_feature, text_features, return_ws=False):
        # image_features 4, b,2352,768
        # image_feature b,p,768
        B = image_feature.shape[0]
        device = image_feature.device
        text_features = text_features.transpose(0,1)
        # class_p = self.class_p.clone()
        # class_p = class_p.to(device)

        img_feature = image_feature.transpose(0,1) # n b d
        # 14 8 768 2352 8 768
        # img_feature = self.decoder_norm(img_feature)
        # text_features = self.decoder_norm(text_features)
        feature,ws = self.decoder(text_features, img_feature, 
            memory_key_padding_mask=None, pos=None, query_pos=None)
        # no attention
        ws_mean=(ws[-4]+ws[-3]+ws[-2]+ws[-1])/4
        out1 = self.dropout_feas(feature)

        # class seperate classifier
        out = out1.transpose(0,1) # b,14,768
        out = out.unsqueeze(-2) # b,14,1,768
        oB = out.shape[0] # b
        oC = out.shape[1] # 14
        out = rearrange(out,'b c l d -> (b c) l d') # b*14, 1, 768
        clas = self.classifier.repeat(oB,1,1,1) # b,14,768,1
        clas = rearrange(clas,'b c d l -> (b c) d l') # b*14, 768,1

        x = torch.bmm(out,clas) # b*14,1,1
        x = rearrange(x,'(b c) e f -> b c (e f)',b=oB,c=oC) # b, 14, 1

        if "exclude_class" in self.config and self.config["exclude_class"]:
            x = x[:,self.keep_class_dim,:]

        if return_ws:
            return ws_mean
        else:
            return x

    @staticmethod
    def _init_weights(module):
        r"""Initialize weights like BERT - N(0.0, 0.02), bias = 0."""

        if isinstance(module, nn.Linear):
            module.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.MultiheadAttention):
            module.in_proj_weight.data.normal_(mean=0.0, std=0.02)
            module.out_proj.weight.data.normal_(mean=0.0, std=0.02)

        elif isinstance(module, nn.Embedding):
            module.weight.data.normal_(mean=0.0, std=0.02)
            if module.padding_idx is not None:
                module.weight.data[module.padding_idx].zero_()