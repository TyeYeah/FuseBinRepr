from math import sqrt
import torch, random,math
import torch.nn as nn
from tqdm import trange, tqdm

from data_set import *
from conf_utils import *
from loss_func import ContrastiveLoss, TripletLoss
from info_nce import InfoNCE, info_nce


class CalculateAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, Q, K, V, mask):
        attention = torch.matmul(Q,torch.transpose(K, -1, -2))
        # use mask
        attention = attention.masked_fill_(mask, -1e9)
        attention = torch.softmax(attention / sqrt(Q.size(-1)), dim=-1)
        attention = torch.matmul(attention,V)
        return attention

class Multi_CrossAttention(nn.Module):

    def __init__(self,hidden_size,all_head_size,head_num):
        super().__init__()
        self.hidden_size    = hidden_size       # dim of input
        self.all_head_size  = all_head_size     # dim of output
        self.num_heads      = head_num          # num of att head
        self.h_size         = all_head_size // head_num

        assert all_head_size % head_num == 0

        # W_Q,W_K,W_V (hidden_size,all_head_size)
        self.linear_q = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_k = nn.Linear(hidden_size, all_head_size, bias=False)
        self.linear_v = nn.Linear(hidden_size, all_head_size, bias=False)
        # self.linear_output = nn.Linear(all_head_size, hidden_size)
        self.linear_output = nn.Linear(all_head_size, all_head_size)

        # normalization
        self.norm = sqrt(all_head_size)

    def print(self):
        print(self.hidden_size,self.all_head_size)
        print(self.linear_k,self.linear_q,self.linear_v)
    
    def forward(self,x,y,attention_mask):
        """
        cross-attention: x for q and k, y for v
        """
        batch_size = x.size(0)
        # (B, S, D) -proj-> (B, S, D) -split-> (B, S, H, W) -trans-> (B, H, S, W)

        # q_s: [batch_size, num_heads, seq_length, h_size]
        q_s = self.linear_q(x).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # k_s: [batch_size, num_heads, seq_length, h_size]
        k_s = self.linear_k(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        # v_s: [batch_size, num_heads, seq_length, h_size]
        v_s = self.linear_v(y).view(batch_size, -1, self.num_heads, self.h_size).transpose(1,2)

        attention_mask = attention_mask.eq(0)

        attention = CalculateAttention()(q_s,k_s,v_s,attention_mask)
        # attention : [batch_size , seq_length , num_heads * h_size]
        attention = attention.transpose(1, 2).contiguous().view(batch_size, -1, self.num_heads * self.h_size)
        
        # output : [batch_size , seq_length , hidden_size]
        output = self.linear_output(attention)

        return output

class TextGraphAlignment(nn.Module):
    """
    From NSP task, now used for ali
    """

    def __init__(self, hidden):
        """
        :param hidden: BERT model output size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, 2)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        # print(x)
        # print("softmax.........",self.linear(x[:, 0]),'\n=================',self.softmax(self.linear(x[:, 0])))
        return self.softmax(self.linear(x[:, 0]))

class GraphMaskingRecovery(nn.Module):
    """
    predicting origin token from masked input sequence
    n-class classification problem, n-class = vocab_size
    """

    def __init__(self, hidden, vocab_size):
        """
        :param hidden: output size of BERT model
        :param vocab_size: total vocab size
        """
        super().__init__()
        self.linear = nn.Linear(hidden, vocab_size)
        self.softmax = nn.LogSoftmax(dim=-1)

    def forward(self, x):
        return self.softmax(self.linear(x))

class PositionalEmbedding(nn.Module):
    def __init__(self, d_model=512, max_len=512):
        super().__init__()

        # Compute the positional encodings once in log space.
        pe = torch.zeros(max_len, d_model).float()
        pe.require_grad = False

        position = torch.arange(0, max_len).float().unsqueeze(1)
        div_term = (torch.arange(0, d_model, 2).float() * -(math.log(10000.0) / d_model)).exp()

        pe[:, 0::2] = torch.sin(position * div_term)
        pe[:, 1::2] = torch.cos(position * div_term)

        pe = pe.unsqueeze(0)
        self.register_buffer('pe', pe)
    def forward(self, x):
        return self.pe[:, :x.size(1)]

class SegmentEmbedding(nn.Embedding):
    def __init__(self, embed_size=512):
        super().__init__(3, embed_size, padding_idx=0)

class InputEmbedding(nn.Module):
    """
    BERT Embedding which is consisted with under features
        1. TokenEmbedding : normal embedding matrix
        2. PositionalEmbedding : adding positional information using sin, cos
        2. SegmentEmbedding : adding sentence segment info, (sent_A:1, sent_B:2)

        sum of all these features are output of BERTEmbedding
    """
    def __init__(self, input_size, embed_size, dropout=0.1):
        """
        :param embed_size: embedding size of token embedding
        :param dropout: dropout rate
        """
        super().__init__()
        # self.position = PositionalEmbedding(d_model=embed_size)
        # self.segment = SegmentEmbedding(embed_size=embed_size)
        self.linear = nn.Linear(input_size, embed_size)
        self.dropout = nn.Dropout(p=dropout)

    def forward(self, sequence, segment_label='torch.tensor([0 for i in range(len(sequence))])'):
        x = self.linear(sequence) # + self.position(sequence) + self.segment(segment_label)
        return self.dropout(x) # overcome overfitting
        return x

class CrossAttFuseMod(nn.Module):
    def __init__(self,params):
        super().__init__()
        self.params = params

        ### cross attention
        if "cross" in self.params.note:
            self.text_embedding_layer = InputEmbedding(params.text_size,params.hidden_size)
            self.graph_embedding_layer = InputEmbedding(params.graph_size,params.hidden_size)

            self.mix_layers_a = nn.ModuleList(
                [Multi_CrossAttention(params.hidden_size,params.hidden_size,params.head_num) for _ in range(params.mix_layer_num)])
            self.mix_layers_b = nn.ModuleList(
                [Multi_CrossAttention(params.hidden_size,params.hidden_size,params.head_num) for _ in range(params.mix_layer_num)])
            
            self.text_embedding_layer2 = InputEmbedding(params.hidden_size, params.all_head_size)
            self.graph_embedding_layer2 = InputEmbedding(params.hidden_size, params.all_head_size)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=params.all_head_size*2, nhead=params.head_num)
            self.self_att = nn.TransformerEncoder(self.encoder_layer, num_layers=params.self_layer_num)

            self.ali = TextGraphAlignment(params.all_head_size*2)
            self.rec = GraphMaskingRecovery(params.all_head_size*2, params.graph_size)

        ### no cross attention
        if "self" in self.params.note:
            self.text_embedding_layer = InputEmbedding(params.text_size,params.hidden_size)
            self.graph_embedding_layer = InputEmbedding(params.graph_size,params.hidden_size)

            self.encoder_layer = nn.TransformerEncoderLayer(d_model=params.all_head_size*2, nhead=params.head_num)
            self.self_att = nn.TransformerEncoder(self.encoder_layer, num_layers=params.self_layer_num)

            self.ali = TextGraphAlignment(params.all_head_size*2)
            self.rec = GraphMaskingRecovery(params.all_head_size*2, params.graph_size)

    def forward(self, text_input, graph_input, attention_mask):

        ### cross attention
        if "cross" in self.params.note:
            text_out = self.text_embedding_layer(text_input) # text_size, graph_size in ; hidden_size out
            graph_out = self.graph_embedding_layer(graph_input) 
            # squeeze 1 dim, form [batch_size , seq_length , hidden_size] to [batch_size , hidden_size]
            text_out, graph_out = text_out.view(text_out.shape[0],self.params.hidden_size), \
                                graph_out.view(graph_out.shape[0],self.params.hidden_size) 
            
            for i in range(self.params.mix_layer_num):
                text_out_tmp = self.mix_layers_a[i](text_out,graph_out,attention_mask)
                graph_out_tmp = self.mix_layers_b[i](graph_out,text_out,attention_mask)
                text_out = text_out_tmp
                graph_out = graph_out_tmp

            mix_out_a = self.text_embedding_layer2(text_out) # hidden_size in, all_head_size out
            mix_out_b = self.graph_embedding_layer2(graph_out) # hidden_size in, all_head_size out
            # no need to sqeeze, but concate [batch_size , seq_length , all_head_size]
            mix_out = torch.cat((mix_out_a, mix_out_b),2) # concate
            output = self.self_att(mix_out) # all_head_size in & out
            ali_cls = self.ali(output)
            rec_emb = self.rec(output)
        
        ### no cross attention
        if "self" in self.params.note:
            text_out = self.text_embedding_layer(text_input) # text_size, graph_size in ; hidden_size out
            graph_out = self.graph_embedding_layer(graph_input) 
            # squeeze 1 dim, form [batch_size , seq_length , hidden_size] to [batch_size , hidden_size]
            text_out, graph_out = text_out.view(text_out.shape[0],self.params.hidden_size), \
                                graph_out.view(graph_out.shape[0],self.params.hidden_size) 
            
            mix_out = torch.cat((text_out, graph_out),1).view(1,1,-1) # concate

            output = self.self_att(mix_out) # all_head_size in & out
            ali_cls = self.ali(output)
            rec_emb = self.rec(output)

        return output, ali_cls, rec_emb

class CrossAttFuseTrainer(object):
    def __init__(self, params):
        self.params = params
        if params.pretrained_path:
            if params.pretrained_path == "auto": # select latest model version
                pretrain_paths = glob.glob(params.output_model_folder+"*")
                if pretrain_paths == []:
                    self.model = CrossAttFuseMod(params)
                    print("model from scratch... ")
                else:
                    pretrain_path = sorted(pretrain_paths, key = os.path.getctime)[-1]
                    self.model = torch.load(pretrain_path)
                    print("model loaded: " + pretrain_path)
            else: # load specified one
                self.model = torch.load(params.pretrained_path)
                print("model loaded: " + params.pretrained_path)
        else:
            self.model = CrossAttFuseMod(params)
            print("model from scratch... ")
        
        self.device = torch.device("cuda:"+self.params.todevice if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)

    def masking(self,vector):
        target = torch.zeros(vector.shape)
        result = torch.zeros(vector.shape) # for not backward through the graph a second time
        for i in range(vector.size(0)):
            for j in range(vector.size(1)):
                dice = random.random()
                if dice < 0.05:
                    target[i][j] = vector[i][j].data
                else:
                    result[i][j] = vector[i][j].data
                # kinda simple
        return result, target

    def train(self):
        self.model.train()
        device = self.device
        fuse_dataset = FuseDataset(self.params.train_csv_folder, self.params.train_text_folder, self.params.train_graph_folder)
        epoch_num = self.params.epoch_num
        # sum of data = iteration * batch
        batch_size = self.params.batch_size
        iteration_num = fuse_dataset.__len__()//batch_size
        print("epoch_num:",epoch_num, "batch_size:",batch_size, "iteration_num:",iteration_num,"dataset length:",fuse_dataset.__len__())

        optimizer = torch.optim.AdamW(self.model.parameters(), lr=self.params.learning_rate, weight_decay=self.params.weight_decay)
        loss_function_ali = nn.CrossEntropyLoss().to(device)
        loss_function_con = ContrastiveLoss().to(device) 
        # loss_function_con = InfoNCE().to(device) # TripletLoss(1.).to(device) 
        # loss_function_con = TripletLoss(1.).to(device) 
        loss_function_rec = nn.HuberLoss().to(device) # nn.HuberLoss().to(device)
        attention_mask = torch.ones(1).to(device)

        makesuredirs("./log"); makesuredirs("./model")
        list_loss_ali = []; list_loss_con = []; list_loss_rec = []; list_loss_accu = []
        loss_ali = torch.tensor(0);loss_con = torch.tensor(0);loss_rec = torch.tensor(0)

        torch.save(self.model,self.params.output_model_folder+self.params.note+'_model.ep0.pt')

        # pairs contrastive loss
        for epoch in tqdm(range(epoch_num), desc="Epochs: ", leave=True):
            ittq = tqdm(range(iteration_num))
            for iter in ittq: 
                ittq.set_description("Iter: ali "+str(round(loss_ali.item(),4))+" CON "+str(round(loss_con.item(),4))+" rec "+str(round(loss_rec.item(),4)))
                accumulated_losses = 0
                tmp_output = 0 # empty tensor
                for bat in range(batch_size):
                    emb1_text, emb1_graph, emb2_text, emb2_graph, label, ll = fuse_dataset.__getitem__(iter*batch_size+bat)
                    emb1_text, emb1_graph, emb2_text, emb2_graph = emb1_text.to(device), emb1_graph.to(device), emb2_text.to(device), emb2_graph.to(device)
                    # print(emb1_graph.shape,"=========================================") # [1,1024]
                    # input("pause")

                    ### rec preparing
                    emb1_graph, target_rec_emb1 = self.masking(emb1_graph)
                    emb1_graph, target_rec_emb1 = emb1_graph.to(device), target_rec_emb1.to(device)
                    emb2_graph, target_rec_emb2 = self.masking(emb2_graph)
                    emb2_graph, target_rec_emb2 = emb2_graph.to(device), target_rec_emb2.to(device)
                    # emb1_graph, target_rec_emb1 = emb1_graph.to(device), emb1_graph.to(device)
                    # emb2_graph, target_rec_emb2 = emb2_graph.to(device), emb1_graph.to(device)

                    # print(emb1_text.shape, emb1_graph.shape, emb2_text.shape, emb2_graph.shape,"---------")
                    output11, ali_cls11, rec_emb11 = self.model(emb1_text, emb1_graph, attention_mask)
                    output22, ali_cls22, rec_emb22 = self.model(emb2_text, emb2_graph, attention_mask)
                    # print(emb1_text)
                    # print(emb1_graph)
                    # print(emb2_text)
                    # print(emb2_graph)
                    # print(output11,output22)
                    # print("++++++++++++++++++++++++++++++")

                    ### loss_ali  #   space alignment classification
                    dice = random.random() # TODO: should throw the dice twice here. 
                    if dice < 0.25:
                        ali_cls = ali_cls11
                        target_ali = torch.tensor([1]).to(device)
                    elif 0.25 <= dice < 0.5:
                        output12, ali_cls12, rec_emb12 = self.model(emb1_text, emb2_graph, attention_mask)
                        ali_cls = ali_cls12
                        target_ali = torch.tensor([0]).to(device)
                    elif 0.5 <= dice < 0.75:
                        output21, ali_cls21, rec_emb21 = self.model(emb2_text, emb1_graph, attention_mask)
                        ali_cls = ali_cls21
                        target_ali = torch.tensor([0]).to(device)
                    else:
                        ali_cls = ali_cls22
                        target_ali = torch.tensor([1]).to(device)
                    # print(ali_cls.shape,target_ali.shape,"=========")
                    loss_ali = loss_function_ali(ali_cls,target_ali)
                    # print("calculate loss_ali:",loss_ali)
                    list_loss_ali.append(loss_ali.item())
                    # list_loss_ali.append(0)

                    ### loss_con  #   contrastive learning
                    # print(output11.shape,output22.shape,"=========")
                    loss_con = loss_function_con(output11,output22,torch.tensor((int(label)+1)/2).to(device))
                    # print(output11,'\n',output22,'\n',label,torch.tensor((int(label)+1)/2).to(device),'\n',loss_con,'\n')
                    # print("calculate loss_con:",loss_con)
                    list_loss_con.append(loss_con.item())

                    ### loss_rec  #   space masked recovery
                    if dice < 0.5:
                        rec_emb = rec_emb11
                        target_rec = target_rec_emb1
                    else:
                        rec_emb = rec_emb22
                        target_rec = target_rec_emb2
                    # print(rec_emb.shape,target_rec.shape,"=========")
                    loss_rec = loss_function_rec(rec_emb.view(self.params.graph_size),target_rec.view(self.params.graph_size))
                    # print("calculate loss_rec:",loss_rec)
                    list_loss_rec.append(loss_rec.item())
                    # list_loss_rec.append(0)

                    # control loss type by --note
                    loss_all = 0
                    if "_ali" in self.params.note:
                        loss_all+=loss_ali
                    if "_rec" in self.params.note:
                        loss_all+=loss_rec
                    if "_con" in self.params.note:
                        loss_all+=loss_con
                    if "_all" in self.params.note:
                        loss_all = loss_con + loss_rec + loss_ali  # +loss_rec # +loss_ali # +loss_rec
                    list_loss_accu.append(loss_all.item())

                    accumulated_losses += loss_all

                accumulated_losses = accumulated_losses/batch_size
                optimizer.zero_grad()
                accumulated_losses.backward()
                optimizer.step()
                # input("pause ...")

            if epoch % 1 == 0:  # draw loss pic every ep
                draw_loss(list_loss_ali, list_loss_con, list_loss_rec, list_loss_accu, self.params.output_log_folder+self.params.note)
                log_content = time.strftime("%Y-%m-%d.%H:%M:%S",time.gmtime())+"\t"+self.params.note+"\tLoss drawed.\n"
                torch.save(self.model,self.params.output_model_folder+self.params.note+'_model.ep'+str(epoch)+'.'+time.strftime("%Y%m%d%H%M%S",time.gmtime())+'.pt')
                log_content += time.strftime("%Y-%m-%d.%H:%M:%S",time.gmtime())+"\t"+self.params.note+"\tModel saved."
                os.system("echo '"+log_content+"'>> "+self.params.output_log_folder+'log.log')

    def evaluate(self):
        self.model.eval()
        device = self.device
        fuse_dataset = FuseDataset(self.params.test_csv_folder, self.params.test_text_folder, self.params.test_graph_folder)
        attention_mask = torch.ones(1,8,1,1).to(device)
    
        # threshold
        threshold = 0.8
        tp=1;tn=1;fp=1;fn=1 ; tt=1;ff=1
        ittq = tqdm(range(fuse_dataset.__len__()))

        # pairs
        for iter in ittq:
            ittq.set_description("tp:"+str(tp)+", tn:"+str(tn)+", fp:"+str(fp)+", fn:"+str(fn))
            emb1_text, emb1_graph, emb2_text, emb2_graph, label, ll = fuse_dataset.__getitem__(iter)
            emb1_text, emb1_graph, emb2_text, emb2_graph = emb1_text.to(device), emb1_graph.to(device), emb2_text.to(device), emb2_graph.to(device)
            
            output1, _, _ = self.model(emb1_text, emb1_graph, attention_mask)
            output2, _, _ = self.model(emb2_text, emb2_graph, attention_mask)
            
            # generate embedding pkl files along with evaluation
            # print(ll,"-------------------------------------------------------------------")
            bin_and_func_1 = ll[0].split("/")[-2:]
            makesuredirs(self.params.output_pkl_folder+self.params.note+'/'+bin_and_func_1[0])
            target_pkl_path_1 = self.params.output_pkl_folder+self.params.note+'/'+bin_and_func_1[0]+'/'+bin_and_func_1[1]
            bin_and_func_2 = ll[2].split("/")[-2:]
            makesuredirs(self.params.output_pkl_folder+self.params.note+'/'+bin_and_func_2[0])
            target_pkl_path_2 = self.params.output_pkl_folder+self.params.note+'/'+bin_and_func_2[0]+'/'+bin_and_func_2[1]
            # print(emb1_text, emb1_graph, emb2_text, emb2_graph)
            with open(target_pkl_path_1,'wb') as pf:
                # print(target_pkl_path_1,",,,,,,,,,,,,,,,,,,,,,,,,,,,,,,,")
                pickle.dump(output1,pf)
            with open(target_pkl_path_2,'wb') as pf:
                # print(target_pkl_path_2,"...............................")
                pickle.dump(output2,pf)
            
            sim = torch.cosine_similarity(output1.view(1,-1),output2.view(1,-1))

            # # debug usage
            # print("==========================\n+                        +\n+         start          +\n+          debug         +\n+                        +\n==========================")
            # print("emb1_text",emb1_text, "\nemb1_graph",emb1_graph, "\nemb2_text",emb2_text, "\nemb2_graph",emb2_graph,"\n++++++++++++++++++++++++++++")
            # print(ll,"++++++++++++++++++++")
            # print("output1",output1,output1.size(),"======\noutput2",output2,output2.size(),"================")
            # print("sim and label",sim,"---",label,"=====")
            # input("pause ...............................")

            if sim>threshold :
                if label == "1":
                    tp+=1
                elif label == "0":
                    fp+=1
                # else:
                #     print("AAAAAAAAAAAAAAAAAAAA",label,type(label))
            elif sim<=threshold:
                if label == "1":
                    fn+=1
                elif label == "0":
                    tn+=1
            #     else:
            #         print("BBBBBBBBBBBBBBBBBBBB",label,type(label))
            # else:
            #     print("CCCCCCCCCCCCCCCCCCCC",label,type(label))
            print(sim,threshold,label)

        stat_banner = self.params.note+" model threshold evaluation \n tp:"+str(tp)+", tn:"+str(tn)+", fp:"+str(fp)+", fn:"+str(fn)+"\n"
        accuracy = (tp+tn)/(tp+tn+fp+fn)
        precision = (tp)/(tp+fp)
        recall = (tp)/(tp+fn)
        f1 = 2*precision*recall / (precision+recall)
        stat_banner += "accuracy:"+str(accuracy)+", precision:"+str(precision)+", recall:"+str(recall)+", f1 score:"+str(f1)+"\n"
        stat_banner += "tt:"+str(tt)+", ff:"+str(ff)+"accuracy:"+str(tt/(tt+ff))+"\n"
        print(stat_banner)
        os.system('echo "'+time.asctime(time.localtime(time.time()))+'\n'+stat_banner+'" >> '+self.params.output_log_folder+'log.log')
        
        # ranking
