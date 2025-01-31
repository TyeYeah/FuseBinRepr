import shutil
import torch, os
import numpy as np
from matplotlib import colors, markers
import matplotlib.pyplot as plt
from sklearn import manifold
from sklearn.metrics import auc, roc_curve
import time
import argparse
import texttable
import multiprocessing, subprocess

def parameter_parser():
    """
    A method to parse up command line parameters. By default it learns on the Watts-Strogatz dataset.
    The default hyperparameters give good results without cross-validation.
    """
    parser = argparse.ArgumentParser(description="Run FuseBinRepr.")
    
    parser.add_argument("--train-csv-folder",
                        nargs="?",
                        default="/root/bcsd/scripts/train/",
	                help="Training csv file folder.")

    parser.add_argument("--test-csv-folder",
                        nargs="?",
                        default="/root/bcsd/scripts/test/",
	                help="Testing csv file folder.")
    
	# default="/mnt/bcsd/asm2vec_cache/asm/",   # dim 200
    # default="/mnt/bcsd/clap_cache/out/",      # dim 768
    # default="/mnt/bcsd/gmn_cache/out/",       # dim 1024
    # default="/mnt/bcsd/safe_cache/output/",   # dim 100
    # default="/mnt/bcsd/sem2vec_cache/gmn/",   # dim 1536
    # default="../PositionDistributionMatters_w/PDM/out/",      # dim 256

    parser.add_argument("--train-text-folder",
                        nargs="?",
                        default="/mnt/bcsd/clap_cache/out/",    # 
	                help="Training texts folder.")

    parser.add_argument("--test-text-folder",
                        nargs="?",
                        default="/mnt/bcsd/clap_cache/out/",    # 
	                help="Testing texts folder.")

    parser.add_argument("--train-graph-folder",
                        nargs="?",
                        default="/mnt/bcsd/gmn_cache/out/",
	                help="Training graphs folder.")

    parser.add_argument("--test-graph-folder",
                        nargs="?",
                        default="/mnt/bcsd/gmn_cache/out/",
	                help="Testing graphs folder.")

    parser.add_argument("--epoch-num",
                        type=int,
                        default=3,
	                help="Number of training epochs. Default is 11.")

    parser.add_argument("--batch-size",
                        type=int,
                        default=32,
	                help="Number of graphs processed per batch. Default is 32.")

    parser.add_argument("--weight-decay",
                        type=float,
                        default=10**-6, # 1e-6
	                help="Weight decay. Default is 10^-6.")

    parser.add_argument("--learning-rate",
                        type=float,
                        default= 0.0001,
	                help="Learning rate. Default is 0.001.")
    
    parser.add_argument("--text-size",
                        type=int,
                        default=768,    # 
	                help="Size of text embedding.")

    parser.add_argument("--graph-size",
                        type=int,
                        default=1024,   # 
	                help="Size of graph embedding.")

    parser.add_argument("--hidden-size",
                        type=int,
                        default=512,
	                help="Dimension of hidden layers. Default is 512.")

    parser.add_argument("--all-head-size",
                        type=int,
                        default=512,
	                help="Dimension of output embedding. Default is 512.")

    parser.add_argument("--head-num",
                        type=int,
                        default=8,
	                help="Number of self-attention layer (Transformer encoder) head. Default is 8.")

    parser.add_argument("--mix-layer-num",
                        type=int,
                        default=3,
	                help="Number of cross-attension layers. Default is 3.")

    parser.add_argument("--self-layer-num",
                        type=int,
                        default=6,
	                help="Number of self-attension layers. Default is 6.")
   
    parser.add_argument("--pretrained-path",
                        nargs="?",
                        default=None,
	                help="Pretrained model path. Default is None, `auto` to get latest in `./model`")
    
    parser.add_argument("--output-model-folder",
                        nargs="?",
                        default="/mnt/bcsd/f_cache/model/",
	                help="Output model folder.")

    parser.add_argument("--output-pkl-folder",
                        nargs="?",
                        default="/mnt/bcsd/f_cache/out/",
	                help="Output pkl folder, for further evaluation.")

    parser.add_argument("--output-log-folder",
                        nargs="?",
                        default="./log/",
	                help="Output log file and statistical chart folder.")

    parser.add_argument("--note",
                        nargs='?',
                        default="",
	                help="Note file name.")
    
    parser.add_argument("--todevice",
                        nargs='?',
                        default="0",
	                help="Specify CUDA/CPU device")


    parser.add_argument("--expmode",
                        nargs='?',
                        default="train",
	                help="Choose the experiment mode. (train or eval)")

    return parser.parse_args()

def tab_printer(args):
    """
    Function to print the logs in a nice tabular format.
    :param args: Parameters used for the model.
    """
    args = vars(args)
    keys = sorted(args.keys())
    t = texttable.Texttable() 
    t.add_rows([["Parameter", "Value"]])
    t.add_rows([[k.replace("_", " ").capitalize(), args[k]] for k in keys])
    print(t.draw())

def validelf(filename):
    magics = [bytes.fromhex('7f454c46')]
    with open(filename, 'rb') as f:
        header = f.read(4)
        return header in magics

def validpkl(filename):
    magics = [bytes.fromhex('80')] # 800495 
    with open(filename, 'rb') as f:
        header = f.read(1) # 3 
        return header in magics

def check_pkl(filename):
    return not is_empty(filename) and validpkl(filename)

def makesuredirs(target_dir):
    if not os.path.exists(target_dir):
        os.makedirs(target_dir)

def removesuredirs(target_dir):
    if os.path.exists(target_dir):
        shutil.rmtree(target_dir)

def is_empty(file_path):
    # with open(file_path, "r", encoding="utf-8") as f:
    #     first_char = f.read(1)
    #     if not first_char:
    #         return True
    # return False
    return os.path.getsize(file_path) == 0  # os.stat(file_path).st_size == 0
def is_exist(file_path):
    # return os.path.exists(file_path)
    return os.path.isfile(file_path)

def calmrr(ll):
    llen = len(ll)
    if llen == 0:
        return 0,0
    res = 0
    rec=0
    for lll in ll:
        res+=1/lll
        if lll ==1:
            rec+=1
    return res/llen,rec/llen

def calstr(s="sss"):
    bias = 100
    if str!=type(s):
        if int == type(s):
            return s+1
        else: return 1
    t = 0
    for i in s:
        t+=ord(i)
    return t+1+bias

def run_command(command):
    process = subprocess.Popen(command, shell=True, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    stdout, stderr = process.communicate()
    print(f"Command: {command}\nOutput: {stdout.decode()}\nError: {stderr.decode()}")

def draw_tsne(X=np.array([[0, 0, 0], [0, 1, 1], [1, 0, 1], [1, 1, 1]]),y=[12,34,56,78], dim_ind = 'i'):
    # X:features    ;   y:labels
    '''X:features and y:labels'''
    X=np.array(X)
    
    yy = [ calstr(i)%148 for i in y]
    # print(yy)
    tsne = manifold.TSNE(n_components=3, init='pca', random_state=501)
    X_tsne = tsne.fit_transform(X)
    print("Org data dimension is {}.Embedded data dimension is {}".format(X.shape[-1], X_tsne.shape[-1]))

    colors_list = []
    for colorrr in colors.cnames.keys():
        colors_list.append(colorrr)
    marker_list = []
    for mark in markers.MarkerStyle.markers.keys():
        marker_list.append(mark)
        
    # visualization
    x_min, x_max = X_tsne.min(0), X_tsne.max(0)
    X_norm = (X_tsne - x_min) / (x_max - x_min)  # normalization
    plt.figure(figsize=(8, 8))

    hist = []
    for i in range(X_norm.shape[0]):
        if y[i] in hist :
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]], marker=marker_list[yy[i]%len(marker_list)])
        else: 
            plt.scatter(X_norm[i, 0], X_norm[i, 1], color=colors_list[yy[i]],label = y[i], marker=marker_list[yy[i]%len(marker_list)])
            hist.append(y[i])
        # plt.text(X_norm[i, 0], X_norm[i, 1], str(y[i]), color=plt.cm.Set1(y[i]), 
        #         fontdict={'weight': 'bold', 'size': 9})
    plt.xticks([])
    plt.yticks([])
    plt.legend()
    plt.show()
    plt.savefig("Tsne-"+str(dim_ind)+".png") # plt.show()
    pass

def calculate_auc(labels, predicts, name_list, note=''):
    # print('--debug-- ',labels,predicts)
    fpr_list = []
    tpr_list = []
    AUC_list = []
    for i in range(len(labels)):
        fpr, tpr, thresholds = roc_curve(labels[i], predicts[i], pos_label=1)
        fpr_list.append(fpr)
        tpr_list.append(tpr)
    for i in range(len(fpr_list)):
        AUC = auc(fpr_list[i], tpr_list[i])
        AUC_list.append(AUC)
        print ("auc"+note+" : ",AUC)

    colors_list = []
    for colorrr in colors.cnames.keys():
        colors_list.append(colorrr)

    plt.figure()
    lw=2
    #plt.figure(figsize(10,10))
    for i in range(len(fpr_list)):
        plt.plot(fpr_list[i],tpr_list[i],color=colors_list[i+10],lw=lw,label='%s ROC curve (area=%0.2f)'%(name_list[i],AUC_list[i]))
    plt.plot([0,1],[0,1],color='navy',lw=lw,linestyle='--')
    plt.xlim([0.0,1.0])
    plt.ylim([0.0,1.05])
    plt.xlabel('False Positive Rate')
    plt.ylabel('True Positive Rate')
    plt.title('ROC Curve')
    plt.legend(loc="lower right")
    #plt.show()
    statistics_file = "./statistics/"
    if not os.path.exists(statistics_file) :
        os.makedirs(statistics_file)
    plt.savefig(statistics_file+"roc"+str(note)+".png")
#=======================================
    return AUC

def draw_loss(list_sac, list_con, list_smr, list_acc, topath="./"):
    x = [i for i in range(len(list_sac))]
    plt.plot(x,list_sac, label="SAC task loss", color='y')
    plt.plot(x,list_con, label="Contrastive loss", color='g')
    plt.plot(x,list_smr, label="SMR task loss", color='b')
    plt.plot(x,list_acc, label="Accumulated loss", color='k')
    plt.xlabel("epoch & iteration")
    plt.ylabel("loss value")
    plt.title("Loss Log")
    plt.legend()
    # plt.show()
    plt.savefig(topath+'loss.'+time.strftime("%Y-%m-%d.%H:%M:%S",time.gmtime())+".png")
    plt.close()