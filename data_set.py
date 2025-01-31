from torch.utils.data import Dataset
import pickle,glob
from tqdm import tqdm

from conf_utils import *

class FuseDataset(Dataset):
    def __init__(self, csv_dir, text_dir, graph_dir):
        self.csv_dir = csv_dir
        self.text_dir = text_dir
        self.graph_dir = graph_dir

        print("loading csv file, text embeddings and graph embeddings...")
        self.csv_paths = glob.glob(self.csv_dir+"*.csv")
        self.text_paths = glob.glob(self.text_dir+"*/*.pkl")
        print("text num:",len(self.text_paths))
        self.graph_paths = glob.glob(self.graph_dir+"*/*.pkl") # pkl or pkl2
        print("graph num:",len(self.graph_paths))
        print("finish loading, preparing...")

        self.list_csvs = []
        for csv_path in self.csv_paths:
            self.prepare_csv_data(csv_path) # gen pair
            # self.gen_tripair(csv_path) # gen sample
        # print(self.__len__(),"=======")
        # input("pause ...")

    def __getitem__(self, idx): # gen pairs
        return self.list_csvs[idx]  #   emb1_text, emb1_graph, emb2_text, emb2_graph, label
    
    def gen_tripair(self,csv_path): # pos1_text, pos1_graph, pos2_text, pos2_graph, pos3_text, pos3_graph
        if 'tri' not in csv_path:
            return 0
        with open(csv_path,"r") as cf:
            lls = cf.readlines()
            sample_num = 0
            lltq = tqdm(lls)

            for ll in lltq:
                lltq.set_description("valid sample num:"+str(sample_num))
                if len(ll.split(",")) !=6: # true log, but not sample, neglect
                    continue
                bin1, func1, bin2, func2, bin3, func3 = ll.strip().split(",")
                bin1, bin2, bin3 = bin1.replace("/","-"), bin2.replace("/","-"), bin3.replace("/","-")
                pkl1_text = self.text_dir+bin1+'/'+func1+'.pkl'
                pkl1_graph = self.graph_dir+bin1+'/'+func1+'.pkl' # pkl or pkl2
                pkl2_text = self.text_dir+bin2+'/'+func2+'.pkl'
                pkl2_graph = self.graph_dir+bin2+'/'+func2+'.pkl' # pkl or pkl2
                pkl3_text = self.text_dir+bin3+'/'+func3+'.pkl'
                pkl3_graph = self.graph_dir+bin3+'/'+func3+'.pkl' # pkl or pkl2
                lll = [pkl1_text,pkl1_graph,pkl2_text,pkl2_graph,pkl3_text,pkl3_graph]
                if pkl1_text not in self.text_paths or pkl1_graph not in self.graph_paths or pkl2_text not in self.text_paths or pkl2_graph not in self.graph_paths or pkl3_text not in self.text_paths or pkl3_graph not in self.graph_paths:
                    # print(lll)
                    # print(pkl1_text not in self.text_paths , pkl1_graph not in self.graph_paths , pkl2_text not in self.text_paths , pkl2_graph not in self.graph_paths, pkl3_text not in self.text_paths , pkl3_graph not in self.graph_paths)
                    # print("in this csv data item, sth pkl not generated")
                    # input()
                    continue
                elif not check_pkl(pkl1_text) or not check_pkl(pkl1_graph) or not check_pkl(pkl2_text) or not check_pkl(pkl2_graph) or not check_pkl(pkl3_text) or not check_pkl(pkl3_graph):  # in case file not exist
                    # print(lll)
                    # print(not check_pkl(pkl1_text) , not check_pkl(pkl1_graph) , not check_pkl(pkl2_text) , not check_pkl(pkl2_graph))
                    # print("in this csv data item, sth pkl is empty, or wrong format")
                    # input()
                    continue
                emb1_text = self.load_text_emb(pkl1_text)
                emb1_graph = self.load_graph_emb(pkl1_graph)
                emb2_text = self.load_text_emb(pkl2_text)
                emb2_graph = self.load_graph_emb(pkl2_graph)
                emb3_text = self.load_text_emb(pkl3_text)
                emb3_graph = self.load_graph_emb(pkl3_graph)
                data_item = (emb1_text, emb1_graph, emb2_text, emb2_graph, emb3_text, emb3_graph, lll)
                self.list_csvs.append(data_item)
                sample_num += 1

        pass # todo. convert from pair.csv, or generate from triplet.csv
    
    def __len__(self):
        return len(self.list_csvs)
    
    def load_text_emb(self, pkl_path):  # get torch.Size([1, x])
        # print(pkl_path)
        emb_text = pickle.load(open(pkl_path,"rb"))
        emb_text = emb_text.view(1, -1)
        return emb_text

    def load_graph_emb(self, pkl_path):   # get torch.Size([1, x])
        # print(pkl_path)
        emb_graph = pickle.load(open(pkl_path,"rb"))
        emb_graph = emb_graph.view(1,-1) 
        return emb_graph

    def prepare_csv_data(self,csv_path):
        if 'tri' in csv_path:
            return 0
        with open(csv_path,"r") as cf:
            lls = cf.readlines()
            sample_num = 0
            lltq = tqdm(lls)
            
            for ll in lltq:
                lltq.set_description("valid sample num:"+str(sample_num))
                if len(ll.split(",")) !=5: # true log, but not sample, neglect
                    continue
                bin1, func1, bin2, func2, label = ll.split(",")
                bin1, bin2 = bin1.replace("/","-"), bin2.replace("/","-")
                pkl1_text = self.text_dir+bin1+'/'+func1+'.pkl'
                pkl1_graph = self.graph_dir+bin1+'/'+func1+'.pkl' # pkl or pkl2
                pkl2_text = self.text_dir+bin2+'/'+func2+'.pkl'
                pkl2_graph = self.graph_dir+bin2+'/'+func2+'.pkl' # pkl or pkl2
                lll = [pkl1_text,pkl1_graph,pkl2_text,pkl2_graph]
                if pkl1_text not in self.text_paths or pkl1_graph not in self.graph_paths or pkl2_text not in self.text_paths or pkl2_graph not in self.graph_paths:
                    print(lll)
                    print(pkl1_text not in self.text_paths , pkl1_graph not in self.graph_paths , pkl2_text not in self.text_paths , pkl2_graph not in self.graph_paths)
                    print("in this csv data item, sth pkl not generated")
                    # input(1)
                    continue
                elif not check_pkl(pkl1_text) or not check_pkl(pkl1_graph) or not check_pkl(pkl2_text) or not check_pkl(pkl2_graph):  # in case file not exist
                    print(lll)
                    print(not check_pkl(pkl1_text) , not check_pkl(pkl1_graph) , not check_pkl(pkl2_text) , not check_pkl(pkl2_graph))
                    print("in this csv data item, sth pkl is empty, or wrong format")
                    # input(2)
                    continue
                emb1_text = self.load_text_emb(pkl1_text)
                emb1_graph = self.load_graph_emb(pkl1_graph)
                emb2_text = self.load_text_emb(pkl2_text)
                emb2_graph = self.load_graph_emb(pkl2_graph)
                data_item = (emb1_text, emb1_graph, emb2_text, emb2_graph, label.strip(), lll)
                self.list_csvs.append(data_item)
                sample_num += 1
