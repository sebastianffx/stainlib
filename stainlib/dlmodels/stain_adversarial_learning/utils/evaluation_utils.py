from sklearn.metrics import *
from PIL import * 
import glob
import sys
from utils_patches import *
from skimage.transform import rescale, resize, downscale_local_mean

#This function returns the best threshold found for given model in the validation path provided 
def evaluate_model_validation(model_mitosis,validation_images_path,perf_meassure='f1', func_model=False):

    list_val_imgs = glob.glob(validation_images_path + 'mitosis/' +  '**.png')
    val_batches_paths = [list_val_imgs[i:i+64] for i in range(0,len(list_val_imgs),64)]
    list_neg_test_imgs = glob.glob(validation_images_path + 'non_mitosis/' +'**.png')
    val_neg_batches_paths  = [list_neg_test_imgs[i:i+64] for i in range(0,len(list_neg_test_imgs),64)]
    
    #Computing probabilities for mitotic patches
    probabilites = []
    for batch in val_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(image)
        if func_model:
            probabilites.append(model_mitosis.predict(np.array(images_batch))[0])
        else:
            probabilites.append(model_mitosis.predict_proba(np.array(images_batch)))

    concat_probs = np.array([j for i in probabilites for j in i])
    #Computing probabilities for non mitotic patches
    neg_probabilites = []
    for batch in val_neg_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(image)
        if func_model:    
            neg_probabilites.append(model_mitosis.predict(np.array(images_batch))[0])
        else:
            neg_probabilites.append(model_mitosis.predict_proba(np.array(images_batch)))
        #print(model_mitosis.predict_classes(np.array(test_batches_paths[0])))
    concat_neg_probs = np.array([j for i in neg_probabilites for j in i])
    gt_labels = np.concatenate((np.ones((len(concat_probs),1)),np.zeros((len(concat_neg_probs),1))),axis=0)
    all_probs = np.concatenate((concat_probs,concat_neg_probs),axis=0)
    fpr, tpr, thresholds = roc_curve(gt_labels, all_probs[:,1], pos_label=1, sample_weight=None, drop_intermediate=True)
    roc_auc = auc(fpr,tpr)
    max_f1 = 0
    best_thres = 0
    for cur_thres in thresholds[1:]:
        #print(cur_thres)
        cur_f1 = f1_score(gt_labels, all_probs[:,1]>cur_thres, average='macro')
        if cur_f1>max_f1:
            #print(max_f1)
            max_f1 = cur_f1
            best_thres = cur_thres
    print("Better model found in validation set with thres: " + str(best_thres) +  "; With f1="+str(max_f1) + "; AUC: " + str(roc_auc))
    #print(gt_labels.shape,all_probs.shape)
    return best_thres,max_f1,roc_auc







#This function returns the f1 score of the test set provided 
def evaluate_model_test(model_mitosis,path_test_images,path_neg_test_images,best_val_thres, return_probs=False, func_model=True):
    list_test_imgs = glob.glob(path_test_images + '**.png')
    test_batches_paths = [list_test_imgs[i:i+64] for i in range(0,len(list_test_imgs),64)]

    list_neg_test_imgs = glob.glob(path_neg_test_images + '**.png')
    test_neg_batches_paths = [list_neg_test_imgs[i:i+64] for i in range(0,len(list_neg_test_imgs),64)]
    #print(len(list_test_imgs),len(list_neg_test_imgs))

    total_fp  = 0 
    probabilites = []
    for batch in test_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(image)    
        if func_model:  
            probabilites.append(model_mitosis.predict(np.array(images_batch))[0])  
        else:
            probabilites.append(model_mitosis.predict_proba(np.array(images_batch)))
    concat_probs = np.array([j for i in probabilites for j in i])
    #print(len(concat_probs))
    total_fp  = 0 
    neg_probabilites = []
    for batch in test_neg_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(image)
        if func_model:  
            neg_probabilites.append(model_mitosis.predict(np.array(images_batch))[0])
        else:
            neg_probabilites.append(model_mitosis.predict_proba(np.array(images_batch)))
    concat_neg_probs = np.array([j for i in neg_probabilites for j in i])
    #print(len(concat_neg_probs))
    gt_labels = np.concatenate((np.ones((len(concat_probs),1)),np.zeros((len(concat_neg_probs),1))),axis=0)
    all_probs = np.concatenate((concat_probs,concat_neg_probs),axis=0)
    print(gt_labels.shape,all_probs.shape)
    fpr, tpr, thresholds = roc_curve(gt_labels, all_probs[:,1], pos_label=1, sample_weight=None, drop_intermediate=True)
    roc_auc = auc(fpr,tpr)
    f1_test = f1_score(gt_labels, all_probs[:,1]>best_val_thres, average='macro')
    print("Measures in test set : "+path_test_images)
    print("F1: " + str(f1_test) + " - AUC: " + str(roc_auc))
    if return_probs:
        return roc_auc, f1_test, all_probs
    else:
        return roc_auc, f1_test



def evaluate_model_test_tma(model_gleason,paths_test_images,paths_neg_test_images,best_val_thres, return_probs=False, func_model=True,internal_tcga_test = False):
    #list_test_imgs = glob.glob(path_test_images + '**.jpg')
    test_batches_paths = [paths_test_images[i:i+64] for i in range(0,len(paths_test_images),64)]

    #list_neg_test_imgs = glob.glob(path_neg_test_images + '**.jpg')
    test_neg_batches_paths = [paths_neg_test_images[i:i+64] for i in range(0,len(paths_neg_test_images),64)]
    #print(len(list_test_imgs),len(list_neg_test_imgs))

    total_fp  = 0 
    probabilites = []
    for batch in test_batches_paths:
        images_batch = []
        for pathimg in batch:
            image_tma = load_image(pathimg)
            if internal_tcga_test: #Taking the central 224 pixels of the 256px TCGA patches
                images_batch.append(center_cropping(image_tma, 224))    
            else: # #Resizing the 750px TMA image patch
                images_batch.append(resize(image_tma,output_shape=(224,224,3)))    
        if func_model:  
            probabilites.append(model_gleason.predict(np.array(images_batch))[0])  
        else:
            probabilites.append(model_gleason.predict_proba(np.array(images_batch)))
    concat_probs = np.array([j for i in probabilites for j in i])
    #print(len(concat_probs))
    total_fp  = 0 
    neg_probabilites = []
    for batch in test_neg_batches_paths:
        images_batch = []
        for pathimg in batch:
            image_tma = load_image(pathimg)
            #image = center_cropping(image_tma, 200)
            images_batch.append(resize(image_tma,output_shape=(224,224,3)))    
        if func_model:  
            neg_probabilites.append(model_gleason.predict(np.array(images_batch))[0])
        else:
            neg_probabilites.append(model_gleason.predict_proba(np.array(images_batch)))
    concat_neg_probs = np.array([j for i in neg_probabilites for j in i])
    #print(len(concat_neg_probs))
    gt_labels = np.concatenate((np.ones((len(concat_probs),1)),np.zeros((len(concat_neg_probs),1))),axis=0)
    all_probs = np.concatenate((concat_probs,concat_neg_probs),axis=0)
    print(gt_labels.shape,all_probs.shape)
    fpr, tpr, thresholds = roc_curve(gt_labels, all_probs[:,1], pos_label=1, sample_weight=None, drop_intermediate=True)
    roc_auc = auc(fpr,tpr)
    f1_test = f1_score(gt_labels, all_probs[:,1]>best_val_thres, average='macro')
    print("Measures in test set : "+paths_test_images[0])
    print("F1: " + str(f1_test) + " - AUC: " + str(roc_auc))
    if return_probs:
        return roc_auc, f1_test, all_probs
    else:
        return roc_auc, f1_test


#This function returns the best threshold found for given model in the validation path provided 
def evaluate_model_validation_TCGA(model_tcga,validation_images_path,perf_meassure='f1', func_model=False):

    list_val_imgs = glob.glob(validation_images_path + 'GP3/' +  '**.png')
    val_batches_paths = [list_val_imgs[i:i+64] for i in range(0,len(list_val_imgs),64)]
    list_neg_test_imgs = glob.glob(validation_images_path + 'GP4/' +'**.png')
    val_neg_batches_paths  = [list_neg_test_imgs[i:i+64] for i in range(0,len(list_neg_test_imgs),64)]

    #Computing probabilities for mitotic patches
    probabilites = []
    for batch in val_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(resize(image,output_shape=(224,224,3)))
            #images_batch.append(image)
        if func_model:
            probabilites.append(model_tcga.predict(np.array(images_batch))[0])
        else:
            probabilites.append(model_tcga.predict_proba(np.array(images_batch)))

    concat_probs = np.array([j for i in probabilites for j in i])
    #Computing probabilities for non mitotic patches
    neg_probabilites = []
    for batch in val_neg_batches_paths:
        images_batch = []
        for pathimg in batch:
            image = load_image(pathimg)
            images_batch.append(resize(image,output_shape=(224,224,3)))    
        if func_model:    
            neg_probabilites.append(model_tcga.predict(np.array(images_batch))[0])
        else:
            neg_probabilites.append(model_tcga.predict_proba(np.array(images_batch)))
        #print(model_tcga.predict_classes(np.array(test_batches_paths[0])))
    concat_neg_probs = np.array([j for i in neg_probabilites for j in i])
    gt_labels = np.concatenate((np.ones((len(concat_probs),1)),np.zeros((len(concat_neg_probs),1))),axis=0)
    all_probs = np.concatenate((concat_probs,concat_neg_probs),axis=0)
    fpr, tpr, thresholds = roc_curve(gt_labels, all_probs[:,1], pos_label=1, sample_weight=None, drop_intermediate=True)
    roc_auc = auc(fpr,tpr)
    max_f1 = 0
    best_thres = 0
    for cur_thres in thresholds[1:]:
        #print(cur_thres)
        cur_f1 = f1_score(gt_labels, all_probs[:,1]>cur_thres, average='macro')
        if cur_f1>max_f1:
            #print(max_f1)
            max_f1 = cur_f1
            best_thres = cur_thres
    #print("Better model found in validation set with thres: " + str(best_thres) +  "; With f1="+str(max_f1) + "; AUC: " + str(roc_auc))
    #print(gt_labels.shape,all_probs.shape)
    return best_thres,max_f1,roc_auc