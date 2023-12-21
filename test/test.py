import argparse
import os
import yaml as yaml
import numpy as np
import json

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader
from sklearn.metrics import roc_auc_score,precision_recall_curve,accuracy_score,confusion_matrix,average_precision_score

from models.image_encoder.CNN_VIT import CNN_VIT
from models.model import MRIBrain
from dataset.transbts_ie_dataset import BraTS

from models.tokenization_bert import BertTokenizer
from transformers import AutoModel
from models.before_fuse import *

import tqdm
import os
from utils import plot_auc, check_pred


def get_tokenizer(tokenizer,target_text):
    
    target_tokenizer = tokenizer(list(target_text), padding='max_length', truncation=True, max_length= 64, return_tensors="pt")
    
    return target_tokenizer

def _get_bert_basemodel(bert_model_name):
    try:
        model = AutoModel.from_pretrained(bert_model_name)#, return_dict=True)
        print("text feature extractor:", bert_model_name)
    except:
        raise ("Invalid model name. Check the config file and pass a BERT model from transformers lybrary")

    for param in model.parameters():
        param.requires_grad = False

    return model

def get_text_features(model,text_list,tokenizer,device,max_length):
    # text_token =  tokenizer(list(text_list),add_special_tokens=True,max_length=max_length,pad_to_max_length=True,return_tensors='pt').to(device=device)
    target_tokenizer = tokenizer(list(text_list), padding='max_length', truncation=True, max_length=max_length,return_tensors="pt").to(device)
    # text_features = model.encode_text(text_token)
    text_features = model(input_ids = target_tokenizer['input_ids'],attention_mask = target_tokenizer['attention_mask'])#(**encoded_inputs)
    text_features = text_features.last_hidden_state[:,0,:]
    # text_features = F.normalize(text_features, dim=-1)
    return text_features

def compute_AUCs(gt, pred, n_class):
    """Computes Area Under the Curve (AUC) from prediction scores.
    Args:
        gt: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          true binary labels.
        pred: Pytorch tensor on GPU, shape = [n_samples, n_classes]
          can either be probability estimates of the positive class,
          confidence values, or binary decisions.
    Returns:
        List of AUROCs of all classes.
    """
    AUROCs = []
    gt_np = gt.cpu().numpy()
    pred_np = pred.cpu().numpy()
    print("n_classes",n_class)
    for i in range(n_class):
        # print(target_class[i])
        cur_gt = gt_np[:,i]
        cur_pred = pred_np[:,i]
        # print("before",set(cur_gt))
        if strict_test:
            Mask = (( cur_gt!= -1) & ( cur_gt != 2)).squeeze()
            cur_gt = cur_gt[Mask]
            cur_pred = cur_pred[Mask]
        else:
            Mask = (cur_gt == -1).squeeze()
            cur_gt[Mask] = cur_pred[Mask]
        
        AUROCs.append(roc_auc_score(cur_gt, cur_pred))
    return AUROCs



def test(args,config):

    device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print("Total CUDA devices: ", torch.cuda.device_count()) 
    torch.set_default_tensor_type('torch.FloatTensor')

    file_key = args.mode+'_file'
    test_datasets = BraTS(config[file_key],config['label_file'],config['report_observe'], mode ='val')
    test_dataloader = DataLoader(
            test_datasets,
            batch_size=config['test_batch_size'],
            num_workers=4,
            pin_memory=True,
            sampler=None,
            shuffle=True,
            collate_fn=None,
            drop_last=True,
        )    
                 
    
    print("Creating book")

    all_target_class = json.load(open(config['disease_order'],'r'))
    target_class = all_target_class.copy()
    # target_class=json.load(open(config['disease_order'],'r'))
    if "exclude_class" in config and config["exclude_class"]:
        keep_class_dim = [target_class.index(i) for i in target_class if i not in config["exclude_classes"] ]
        target_class = [target_class[i] for i in keep_class_dim]
        keep_class_dim = [all_target_class.index(i) for i in all_target_class if i not in config["exclude_classes"] ]
        all_target_class = [target_class[i] for i in keep_class_dim]
    json_book = json.load(open(config['disease_book'],'r'))
    json_order=json.load(open(config['disease_order'],'r'))
    disease_book = [json_book[i] for i in json_order]
    # tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    # disease_book_tokenizer = get_tokenizer(tokenizer,disease_book).to(device)

    tokenizer = BertTokenizer.from_pretrained(config['text_encoder'])
    text_encoder = _get_bert_basemodel(config['text_encoder']).to(device)
    text_features = get_text_features(text_encoder,disease_book,tokenizer,device,max_length=256)

    image_encoder = CNN_VIT(img_dim_x=config['input_W'], img_dim_y=config['input_H'], img_dim_z=config['input_D'],embedding_dim=config['d_model'],_conv_repr=True, _pe_type="learned")

    device_ids = [i for i in range(torch.cuda.device_count())]
    image_encoder = nn.DataParallel(image_encoder, device_ids) 
    image_encoder = image_encoder.cuda(device=device_ids[0])
    
    print("Creating model")
    model = MRIBrain(config)

    model = nn.DataParallel(model, device_ids = [i for i in range(torch.cuda.device_count())])
    model = model.to(device)

    print('Load model from checkpoint:',args.model_path)
    checkpoint = torch.load(args.model_path,map_location='cpu') 
    state_dict = checkpoint['model']          
    model.load_state_dict(state_dict)

    image_encoder.load_state_dict(checkpoint['image_encoder'])
    

    # initialize the ground truth and output tensor
    gt = torch.FloatTensor()
    pred = torch.FloatTensor()
    fids = []

    print("Start testing")
    model.eval()

    pred_name = args.model_path.split('/')[-1].split('.')[0]
    pred_name = '_'.join([args.model_path.split('/')[-2], pred_name])
    print("pred_name",pred_name)
    rootdir = "/home/ps/leijiayu/CODE/MedKLIP/Test_Set_Classification_6thself/preds/"+pred_name+main_name+"/"

    if os.path.exists(rootdir+"pred_"+args.mode+".npy"):
        print(rootdir+"pred_"+args.mode+".npy")
        pred = np.load(rootdir+"pred_"+args.mode+".npy")
        gt = np.load(rootdir+"gt_"+args.mode+".npy")
        fids = np.load(rootdir+"fids_"+args.mode+".npy")
        gt = torch.tensor(gt)
        pred = torch.FloatTensor(pred)
    else:
        for i, sample in enumerate(test_dataloader):
            images = sample['image']  # [(b,x,y,z),(b,x,y,z)]
            label = sample['label']

            if "exclude_class" in config and config["exclude_class"]:
                label = label[:,keep_class_dim]

            gt = torch.cat((gt, label), 0)

            B = label.shape[0]
            
            cur_text_features = text_features.unsqueeze(0).repeat(B,1,1)

            fuse_image_feature, _ = image_encoder(images)
            
            # before fuse
            
            #input_image = image.to(device,non_blocking=True)  
            with torch.no_grad():
                pred_class = model(fuse_image_feature,cur_text_features, return_ws=False) #batch_size,num_class,1
                # pred_class = F.softmax(pred_class.reshape(-1,2)).reshape(-1,len(target_class),2)
                # pred_class = pred_class[:,:-1,1]
                # pred_class = pred_class[:,:,1]
        
                pred_class = torch.sigmoid(pred_class.reshape(-1,1)).reshape(-1,len(all_target_class))

                pred = torch.cat((pred, pred_class.detach().cpu()), 0)
                fids += sample["fid"]
                print("fids",pred.shape,gt.shape)
                # np.save("/home/ps/leijiayu/CODE/MedKLIP/Test_Set_Classification_6thself/preds/"+pred_name+"_fids.npy",np.array(fids))
        
        os.makedirs(rootdir,exist_ok=True)
        np.save(rootdir+"pred_"+args.mode+".npy",pred.numpy())
        np.save(rootdir+"gt_"+args.mode+".npy",gt.numpy())
        np.save(rootdir+"fids_"+args.mode+".npy",np.array(fids))

    # print("pred.shape",pred)
    # AUROCs = compute_AUCs(gt, pred,len(target_class))
    AUROCs=[]
    max_f1s = []
    accs = []
    precisions=[]
    recalls=[]
    tns,fps,fns,tps = [],[],[],[]
    threshs= []
    aps = []
    #for i in range(len(target_class)-1):
    for i in range(len(target_class)):   
        gt_np = gt[:, i].numpy()
        pred_np = pred[:, i].numpy()
        Mask = (( gt_np!= -1) & ( gt_np != 2)).squeeze()
        cur_gt = gt_np[Mask]
        cur_pred = pred_np[Mask]
        precision, recall, thresholds = precision_recall_curve(cur_gt, cur_pred)
        numerator = 2 * recall * precision # dot multiply for list
        denom = recall + precision
        f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
        max_f1 = np.max(f1_scores)
        max_f1_thresh = thresholds[np.argmax(f1_scores)]

        if strict_test:
            AUROCs.append(roc_auc_score(cur_gt, cur_pred))
            aps.append(average_precision_score(cur_gt,cur_pred))
            threshs.append(max_f1_thresh)
            precisions.append(precision[np.argmax(f1_scores)])
            recalls.append(recall[np.argmax(f1_scores)])
            max_f1s.append(max_f1)
            accs.append(accuracy_score(cur_gt, cur_pred>max_f1_thresh))
            pred_label = cur_pred >= max_f1_thresh
            tn,fp,fn,tp = confusion_matrix(cur_gt, pred_label).ravel()
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
            plot_auc(cur_gt, cur_pred, target_class[i], rootdir, args.mode)
        else:
            cur_gt = gt_np
            cur_pred = pred_np
            pred_label = cur_pred >= max_f1_thresh
            Mask = (cur_gt == -1).squeeze()
            cur_gt[Mask] = pred_label[Mask]
            precision, recall, thresholds = precision_recall_curve(cur_gt, cur_pred)
            numerator = 2 * recall * precision # dot multiply for list
            denom = recall + precision
            f1_scores = np.divide(numerator, denom, out=np.zeros_like(denom), where=(denom!=0))
            max_f1 = np.max(f1_scores)
            max_f1_thresh = thresholds[np.argmax(f1_scores)]
            aps.append(average_precision_score(cur_gt,cur_pred))
            threshs.append(max_f1_thresh)
            precisions.append(precision[np.argmax(f1_scores)])
            recalls.append(recall[np.argmax(f1_scores)])
            max_f1s.append(max_f1)
            accs.append(accuracy_score(cur_gt, cur_pred>max_f1_thresh))
            pred_label = cur_pred >= max_f1_thresh
            tn,fp,fn,tp = confusion_matrix(cur_gt, pred_label).ravel()
            tns.append(tn)
            fps.append(fp)
            fns.append(fn)
            tps.append(tp)
            plot_auc(cur_gt, cur_pred, target_class[i], rootdir, args.mode)
            AUROCs.append(roc_auc_score(cur_gt, cur_pred))
    
    check_pred(target_class,np.array(fids),threshs,pred,gt,strict_test,rootdir+"result_"+args.mode+".xlsx")

    print('The average AUROC is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(AUROCs[:]).mean()))
    for i in range(len(target_class)):
        print('The AUROC of {} is {}'.format(target_class[i], AUROCs[i]))

    print('The average f1 is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(max_f1s[:]).mean()))
    for i in range(len(target_class)):
        print('The f1 of {} is {}'.format(target_class[i], max_f1s[i]))
    
    print('The average ap is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(aps[:]).mean()))
    for i in range(len(target_class)):
        print('The ap of {} is {}'.format(target_class[i], aps[i]))
    
    print('The average acc is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(accs[:]).mean()))
    for i in range(len(target_class)):
        print('The acc of {} is {}'.format(target_class[i], accs[i]))

    # for i in range(len(target_class)-1):
    print('The average recall is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(recalls[:]).mean()))
    for i in range(len(target_class)):
        print('The recall of {} is {}'.format(target_class[i], recalls[i]))
    print('The average precision is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(precisions[:]).mean()))
    for i in range(len(target_class)):
        print('The precision of {} is {}'.format(target_class[i], precisions[i]))
    #for i in range(len(target_class)-1):
    print('The average thresh is {AUROC_avg:.4f}'.format(AUROC_avg=np.array(threshs[:]).mean()))
    for i in range(len(target_class)):
        print('The thresh of {} is {}'.format(target_class[i], threshs[i]))
        
    # f1_avg = np.array(max_f1s).mean()    
    # acc_avg = np.array(accs).mean()
    # print("all ACC", accs)
    # print('The average f1 is {F1_avg:.4f}'.format(F1_avg=f1_avg))
    # print('The average ACC is {ACC_avg:.4f}'.format(ACC_avg=acc_avg))
    
    
if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    # /home/ps/leijiayu/CODE/MedKLIP/outputdir_34_before_transbts_ie_15dis_lisong_maskdwi/config.yaml
    # /home/ps/leijiayu/CODE/MedKLIP/Test_Set_Classification_6thself/data_file_external/lisong_ex_1.yaml
    parser.add_argument('--config', default='/home/ps/leijiayu/CODE/MedKLIP/Test_Set_Classification_6thself/data_file_external/lisong_ex_1.yaml')
    parser.add_argument('--model_path', default='/home/ps/leijiayu/CODE/MedKLIP/outputdir_34_before_transbts_ie_15dis_lisong/best_val.pth')
    parser.add_argument('--device', default='cuda')
    parser.add_argument('--gpu', type=str,default='1,2,3,0', help='gpu')
    parser.add_argument('--mode', type=str, default='test')
    args = parser.parse_args()

    config = yaml.load(open(args.config, 'r'), Loader=yaml.Loader)

    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu
    if args.gpu != '-1':
        torch.cuda.current_device()
        torch.cuda._initialized = True
    
    strict_test = True
    main_name = "_nostrict" if not strict_test else ""

    test(args, config)