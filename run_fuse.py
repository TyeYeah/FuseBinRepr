from math import sqrt
import torch
import torch.nn as nn

from fuse_mod import *


def CrossAttTest():
    layer = Multi_CrossAttention(768,768,8)
    layer.print()

    model1_output = torch.ones(1,768)
    model2_output = torch.zeros(1,768)
    attention_mask = torch.ones(1) # ones -> no mask, zeros -> mask
    input1, input2 = model1_output, model2_output
    target = torch.ones(768)
    target = torch.tensor(1)
    
    learning_rate = 10**-2
    weight_decay = 1e-4

    layer.train()
    optimizer = torch.optim.Adam(layer.parameters(), lr=learning_rate, weight_decay=weight_decay)
    loss_function = ContrastiveLoss(1.) # nn.CrossEntropyLoss()
    output = layer(input1, input2, attention_mask) 
    loss = loss_function(output[0][0], output[0][0], target)
    optimizer.zero_grad()
    loss.backward()
    optimizer.step()

    layer.eval()
    cross_output = layer(model1_output,model2_output,attention_mask)
    print(cross_output.shape)
    # print(cross_output)

def TrmEncTest():    
    model1_output = torch.ones(768)
    model2_output = torch.zeros(768)
    attention_mask = torch.ones(768) # ones -> no mask, zeros -> mask
    input1, input2 = model1_output, model2_output
    target = torch.ones(768)
    target = torch.tensor([[[i+0.1 for i in range(768) ]]])
    torch.rand(2,2,768)
    encoder_layer = nn.TransformerEncoderLayer(d_model=768, nhead=8)
    transformer_encoder = nn.TransformerEncoder(encoder_layer, num_layers=6)
    encoder_out = transformer_encoder(target) # 3 dimensions
    print('encoder_out',encoder_out.shape)
    
def PosEmbTest():
    pe = PositionalEmbedding()
    se = SegmentEmbedding()
    nl = nn.Linear(12,34)

    input = torch.LongTensor([[1,2,0]])
    input2 = torch.randn(1,12)
    output = se(input)
    output2 = nl(input2)
    print(input,output,input.shape,input.size(0),output.shape,output.squeeze().shape)
    print(input2,output2,input2.shape,output2.shape)

def CrossAttFusionTest():
    args = parameter_parser()
    tab_printer(args)
    args.text_size = 512
    args.graph_size = 512
    
    model = CrossAttFuseMod(args)
    lang_in = torch.randn(1,512)
    graph_in = torch.randn(1,512)

    attention_mask = torch.ones(1)
    fusion,_,_ = model(lang_in, graph_in, attention_mask)
    print(fusion.shape)

def CrossAttTrain(args):
    fuse_trainer = CrossAttFuseTrainer(args)
    fuse_trainer.train()

def CrossAttEval(args):
    fuse_trainer = CrossAttFuseTrainer(args)
    fuse_trainer.evaluate()
    pass
        
''' training commands
    # ### clap + gmn 
    # commands = [
    # "python run_fuse.py --note clap_gmn_self_ali --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_self_all --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",

    # "python run_fuse.py --note clap_gmn_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024",
    # "python run_fuse.py --note clap_gmn_cross_all --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024"
    # ]

    # ### safe + gmn 
    # commands = [
    # "python run_fuse.py --note safe_gmn_self_ali --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_self_all --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",

    # "python run_fuse.py --note safe_gmn_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024",
    # "python run_fuse.py --note safe_gmn_cross_all --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024"
    # ]

    # ### safe + sem2vec 
    # commands = [
    # "python run_fuse.py --note safe_sem_self_ali --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_self_all --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",

    # "python run_fuse.py --note safe_sem_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536",
    # "python run_fuse.py --note safe_sem_cross_all --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536"
    # ]

    # ### clap + sem2vec 
    # commands = [
    # "python run_fuse.py --note clap_sem_self_ali --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_self_all --expmode train --epoch-num 1 --todevice=0 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",

    # "python run_fuse.py --note clap_sem_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536",
    # "python run_fuse.py --note clap_sem_cross_all --expmode train --epoch-num 1 --todevice=1 --train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536"
    # ]

    ############################# and reverse text and graph modalities ....................., another four rounds
    # ### clap + gmn 
    # commands = [
    # "python run_fuse.py --note gmn_clap_self_ali --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_self_all --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",

    # "python run_fuse.py --note gmn_clap_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024",
    # "python run_fuse.py --note gmn_clap_cross_all --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024"
    # ]

    # ### safe + gmn 
    # commands = [
    # "python run_fuse.py --note gmn_safe_self_ali --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_self_all --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",

    # "python run_fuse.py --note gmn_safe_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024",
    # "python run_fuse.py --note gmn_safe_cross_all --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024"
    # ]

    # ### safe + sem2vec 
    # commands = [
    # "python run_fuse.py --note sem_safe_self_ali --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_self_all --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",

    # "python run_fuse.py --note sem_safe_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536",
    # "python run_fuse.py --note sem_safe_cross_all --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536"
    # ]

    # ### clap + sem2vec 
    # commands = [
    # "python run_fuse.py --note sem_clap_self_ali --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_ali_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_rec_con --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_ali_rec --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_self_all --expmode train --epoch-num 1 --todevice=0 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",

    # "python run_fuse.py --note sem_clap_cross_ali --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_ali_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_rec_con --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_ali_rec --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536",
    # "python run_fuse.py --note sem_clap_cross_all --expmode train --epoch-num 1 --todevice=1 --train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536"
    # ]

'''
def cmdtrain(args): # train models in parallel, gpu memory limited


    commands_template = "python run_fuse.py --note placeholder --expmode train --epoch-num 0 "

    note_types = [ "clap_sem", "sem_clap", "safe_sem", "sem_safe", "clap_gmn", "gmn_clap", "safe_gmn", "gmn_safe",]
    note_archs = ["_self", "_cross"]
    note_losss = ["_ali", "_rec", "_con", "_ali_rec", "_ali_con", "_rec_con", "_all"]
    notes_list = []
    for note_type in note_types:
        for note_arch in note_archs:
            for note_loss in note_losss:
                note_tmp = note_type+note_arch+note_loss
                notes_list.append(note_tmp)
    # print(notes_list,len(notes_list))

    iter = 0
    commands = []
    for note in notes_list:
        cmd_tmp = commands_template.replace("placeholder",note)
        if "self" in note:
            cmd_tmp+=" --todevice=0 "
        if "cross" in note:
            cmd_tmp+=" --todevice=1 "

        if "clap_gmn" in note:
            cmd_tmp+="--train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024"
        elif "gmn_clap" in note:
            cmd_tmp+="--train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024"
        elif "safe_gmn" in note:
            cmd_tmp+="--train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024"
        elif "gmn_safe" in note:
            cmd_tmp+="--train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024"
        elif "clap_sem" in note:
            cmd_tmp+="--train-text-folder /mnt/bcsd/clap_cache/out/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536"
        elif "sem_clap" in note:
            cmd_tmp+="--train-graph-folder /mnt/bcsd/clap_cache/out/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536"
        elif "safe_sem" in note:
            cmd_tmp+="--train-text-folder /mnt/bcsd/safe_cache/output/ --train-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536"
        elif "sem_safe" in note:
            cmd_tmp+="--train-graph-folder /mnt/bcsd/safe_cache/output/ --train-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536"
        # print(cmd_tmp)
        # iter+=1
        # input(str(iter))
        commands.append(cmd_tmp)
    # print(commands,len(commands))

    max_workers = 10 # every kind has 7 types to run, 24g graphics card memory can only support 10 each time
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(run_command, commands)

        for output in results:
            print("Command output:", output)

    print("All training commands have been executed.")

def cmdeval(args):  # run evaluation in parallel

    commands_template = "python run_fuse.py --note placeholder --expmode eval --pretrained-path /mnt/bcsd/f_cache/model/placeholder_model.ep0.pt"

    note_types = ["clap_gmn", "gmn_clap", "safe_gmn", "gmn_safe", "clap_sem", "sem_clap", "safe_sem", "sem_safe"]
    # note_types = ["clap_sem", "sem_clap", "safe_sem", "sem_safe"]
    note_archs = ["_self", "_cross"]
    note_losss = ["_ali", "_rec", "_con", "_ali_rec", "_ali_con", "_rec_con", "_all"]
    notes_list = []
    for note_type in note_types:
        for note_arch in note_archs:
            for note_loss in note_losss:
                note_tmp = note_type+note_arch+note_loss
                notes_list.append(note_tmp)
    # print(notes_list,len(notes_list))

    iter = 0
    commands = []
    for note in notes_list:
        cmd_tmp = commands_template.replace("placeholder",note)
        if "self" in note:
            cmd_tmp+=" --todevice=0 "
        if "cross" in note:
            cmd_tmp+=" --todevice=1 "

        if "clap_gmn" in note:
            cmd_tmp+="--test-text-folder /mnt/bcsd/clap_cache/out/ --test-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 768 --graph-size 1024"
        elif "gmn_clap" in note:
            cmd_tmp+="--test-graph-folder /mnt/bcsd/clap_cache/out/ --test-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 768 --text-size 1024"
        elif "safe_gmn" in note:
            cmd_tmp+="--test-text-folder /mnt/bcsd/safe_cache/output/ --test-graph-folder /mnt/bcsd/gmn_cache/out/ --text-size 100 --graph-size 1024"
        elif "gmn_safe" in note:
            cmd_tmp+="--test-graph-folder /mnt/bcsd/safe_cache/output/ --test-text-folder /mnt/bcsd/gmn_cache/out/ --graph-size 100 --text-size 1024"
        elif "clap_sem" in note:
            cmd_tmp+="--test-text-folder /mnt/bcsd/clap_cache/out/ --test-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 768 --graph-size 1536"
        elif "sem_clap" in note:
            cmd_tmp+="--test-graph-folder /mnt/bcsd/clap_cache/out/ --test-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 768 --text-size 1536"
        elif "safe_sem" in note:
            cmd_tmp+="--test-text-folder /mnt/bcsd/safe_cache/output/ --test-graph-folder /mnt/bcsd/sem2vec_cache/gmn/ --text-size 100 --graph-size 1536"
        elif "sem_safe" in note:
            cmd_tmp+="--test-graph-folder /mnt/bcsd/safe_cache/output/ --test-text-folder /mnt/bcsd/sem2vec_cache/gmn/ --graph-size 100 --text-size 1536"
        # print(cmd_tmp)
        # iter+=1
        # input(str(iter))
        commands.append(cmd_tmp)
    # print(commands,len(commands))


    ### run in parallel
    max_workers = 10
    with multiprocessing.Pool(processes=max_workers) as pool:
        results = pool.imap_unordered(run_command, commands)

        for output in results:
            print("Command output:", output)

    print("All evaluation commands have been executed.")

    # for cmd in commands:
    #     run_command(cmd)

if __name__ == "__main__":
    # CrossAttTest()
    # TrmEncTest()
    # PosEmbTest()
    # CrossAttFusionTest()

    args = parameter_parser()
    tab_printer(args)
    if args.expmode == "train":
        CrossAttTrain(args)
    elif args.expmode == "eval": 
        CrossAttEval(args)
    elif args.expmode == "cmdrun":
        cmdtrain(args)
        cmdeval(args)
    else:
        print("expmode???")