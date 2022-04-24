from Layers import *
import torch_geometric.nn as pyg_nn
from torch_geometric.data import Data
import scipy.sparse as sp



class GraphCNN(nn.Module):
    def __init__(self,num_stock, d_market,d_news,out_c,d_hidden , hidn_rnn , hid_c, dropout ,alpha=0.2,alpha1=0.0054,t_mix=1,n_layeres=2,n_heads=1):##alpha1 denotes the normalized threshold
        super(GraphCNN, self).__init__()
        self.t_mix=t_mix
        self.dropout=dropout
        self.num_stock=num_stock
        if  self.t_mix == 0: # concat
            self.GRUs_s = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
            self.GRUs_r = Graph_GRUModel(num_stock, d_market + d_news, hidn_rnn)
        elif self.t_mix == 1: # all_tensor
             self.tensor = Graph_Tensor(num_stock,d_hidden,d_market,d_news)
             self.GRUs_s = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
             self.GRUs_r = Graph_GRUModel(num_stock, d_hidden, hidn_rnn)
        self.gcs=nn.ModuleList()
        self.project=nn.Sequential(
            nn.Linear(hidn_rnn,hid_c),
            nn.Tanh(),
            nn.Linear(hid_c,hid_c,bias=False))
        self.gcs=SMPLayer(hidn_rnn, hid_c,5,3,n_layers=1,dropout=0.2,hgt_layer=1)
        self.attentions = Graph_Attention(hidn_rnn, hid_c, dropout=dropout, alpha=alpha,alpha1=alpha1, residual=True, concat=True)
        self.X2Os = Graph_Linear(num_stock, hidn_rnn+hid_c , out_c, bias = True)
        self.reset_parameters()

    def reset_parameters(self):
        reset_parameters(self.named_parameters)
    def get_relation(self,x_numerical, x_textual):
        x_r = self.tensor(x_numerical, x_textual)
        x_r = self.GRUs_r(x_r)
        relation = torch.stack([att.get_relation(x_r) for att in self.attentions])
        return relation

    def forward(self, x_market,x_news,edge_list,inter_metric,device1):
        if self.t_mix == 0:
            x_s = torch.cat([x_market, x_news], dim=-1)
            x_r = torch.cat([x_market, x_news], dim=-1)
        elif self.t_mix == 1:
            x_s = self.tensor(x_market, x_news)
            x_r = self.tensor(x_market, x_news)
        x_s = self.GRUs_s(x_s)
        x_r = self.GRUs_r(x_r)
        x = (inter_metric @ x_s)  # Generate the initial features of executive nodes
        edge_index=[]
        edge_dict={}
        edge_type=[]
        edge_index1=[]
        #the explicit firm relations
        for eg in ['I','B','S','SC']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
            edge_index1 += edge_list[eg]
        ##considering the implicit relation
        edge_index1=torch.LongTensor(edge_index1).transpose(0,1)
        edge_index1=edge_index1.cuda(device=device1)
        row,col=edge_index1.cpu()
        cc=torch.ones(row.shape[0]).cpu()
        d=1-(sp.coo_matrix((cc, (row, col)), shape=(self.num_stock, self.num_stock))).toarray()
        d[d<1]=0
        c = torch.from_numpy(d).cuda()
        c_adj= self.attentions(x_s, c)
        edge_index+=c_adj.tolist()
        company_list=c_adj.tolist()
        edge_type+=[torch.ones(len(company_list)) * 4]
        edge_type = torch.cat(edge_type, dim=0).cuda(device=device1)


        c_graph=[x_s,torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        edge_index=[]
        edge_dict={}
        edge_type=[]
        ####the relations between firm and executives
        for eg in ['FS','FE']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
        edge_type=torch.cat(edge_type,dim=0).cuda(device=device1)
        t_graph=[torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        edge_index=[]
        edge_dict={}
        edge_type=[]
        ##the explicit executives relatons
        for eg in ['R','C','CL']:
            if eg not in edge_dict:
                edge_dict[eg] = len(edge_dict)
            edge_index += edge_list[eg]
            edge_type += [torch.ones(len(edge_list[eg])) * edge_dict[eg]]
        edge_type=torch.cat(edge_type,dim=0).cuda(device=device1)
        p_graph=[x,torch.LongTensor(edge_index).transpose(0,1).cuda(device=device1),edge_type]
        x = F.dropout(x, self.dropout, training=self.training)
        x_c,x_p=self.gcs(c_graph,p_graph,t_graph)
        x_0 = torch.cat([x_s, x_c], dim=1)
        x_0 = F.elu(self.X2Os(x_0))
        out = F.log_softmax(x_0, dim=1)

        return out
