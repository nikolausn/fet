import math

import torch
import torch.nn as nn
from allennlp.modules.elmo import Elmo
from torch.nn import functional as F
from transformers import *

def token_lens_to_idxs(token_lens):
    """Map token lengths to a word piece index matrix (for torch.gather) and a
    mask tensor.
    For example (only show a sequence instead of a batch):

    token lengths: [1,1,1,3,1]
    =>
    indices: [[0,0,0], [1,0,0], [2,0,0], [3,4,5], [6,0,0]]
    masks: [[1.0, 0.0, 0.0], [1.0, 0.0, 0.0], [1.0, 0.0, 0.0],
            [0.33, 0.33, 0.33], [1.0, 0.0, 0.0]]

    Next, we use torch.gather() to select vectors of word pieces for each token,
    and average them as follows (incomplete code):

    outputs = torch.gather(bert_outputs, 1, indices) * masks
    outputs = bert_outputs.view(batch_size, seq_len, -1, self.bert_dim)
    outputs = bert_outputs.sum(2)

    :param token_lens (list): token lengths.
    :return: a index matrix and a mask tensor.
    """
    max_token_num = max([len(x) for x in token_lens])
    max_token_len = max([max(x) for x in token_lens])
    idxs, masks = [], []
    for seq_token_lens in token_lens:
        seq_idxs, seq_masks = [], []
        offset = 0
        for token_len in seq_token_lens:
            seq_idxs.extend([i + offset for i in range(token_len)]
                            + [-1] * (max_token_len - token_len))
            seq_masks.extend([1.0 / token_len] * token_len
                             + [0.0] * (max_token_len - token_len))
            offset += token_len
        seq_idxs.extend([-1] * max_token_len * (max_token_num - len(seq_token_lens)))
        seq_masks.extend([0.0] * max_token_len * (max_token_num - len(seq_token_lens)))
        idxs.append(seq_idxs)
        masks.append(seq_masks)
    return idxs, masks, max_token_num, max_token_len

class HFet(nn.Module):

    def __init__(self,
                 label_size,
                 elmo_option,
                 elmo_weight,
                 elmo_dropout=.5,
                 repr_dropout=.2,
                 dist_dropout=.5,
                 latent_size=0,
                 svd=None,
                 ):
        super(HFet, self).__init__()
        #self.bert_tokenizer = AutoTokenizer.from_pretrained('allenai/scibert_scivocab_uncased')
        #self.bert_model = AutoModel.from_pretrained('allenai/scibert_scivocab_uncased')
        #self.bert_model.cuda(device=2)

        self.bert_model =  BertModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True)
        self.bert_model.cuda(device=2)

        #self.bert_model_cpu =  BertModel.from_pretrained('allenai/scibert_scivocab_uncased', output_hidden_states = True)

        
        self.label_size = label_size
        #self.elmo = Elmo(elmo_option, elmo_weight, 1,
        #                 dropout=elmo_dropout)
        #self.elmo_dim = self.elmo.get_output_dim()
        self.elmo_dim = 768

        self.attn_dim = 1
        self.attn_inner_dim = self.elmo_dim
        # Mention attention
        self.men_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.men_attn_linear_o = nn.Linear(self.attn_inner_dim, self.attn_dim, bias=False)
        # Context attention
        self.ctx_attn_linear_c = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_m = nn.Linear(self.elmo_dim, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_d = nn.Linear(1, self.attn_inner_dim, bias=False)
        self.ctx_attn_linear_o = nn.Linear(self.attn_inner_dim,
                                        self.attn_dim, bias=False)
        # Output linear layers
        self.repr_dropout = nn.Dropout(p=repr_dropout)
        self.output_linear = nn.Linear(self.elmo_dim * 2, label_size, bias=False)

        # SVD
        if svd:
            svd_mat = self.load_svd(svd)
            self.latent_size = svd_mat.size(1)
            self.latent_to_label.weight = nn.Parameter(svd_mat, requires_grad=True)
            self.latent_to_label.weight.requires_grad = False
        elif latent_size == 0:
            self.latent_size = int(math.sqrt(label_size))
        else:
            self.latent_size = latent_size
        self.latent_to_label = nn.Linear(self.latent_size, label_size,
                                         bias=False)
        self.latent_scalar = nn.Parameter(torch.FloatTensor([.1]))
        self.feat_to_latent = nn.Linear(self.elmo_dim * 2, self.latent_size,
                                        bias=False)
        # Loss function
        self.criterion = nn.MultiLabelSoftMarginLoss()
        self.mse = nn.MSELoss()
        # Relative position (distance)
        self.dist_dropout = nn.Dropout(p=dist_dropout)

    def load_svd(self, path):
        print('Loading SVD matrices')
        u_file = path + '-Ut'
        s_file = path + '-S'
        with open(s_file, 'r', encoding='utf-8') as r:
            s_num = int(r.readline().rstrip())
            mat_s = [[0] * s_num for _ in range(s_num)]
            for i in range(s_num):
                mat_s[i][i] = float(r.readline().rstrip())
        mat_s = torch.FloatTensor(mat_s)

        with open(u_file, 'r', encoding='utf-8') as r:
            mat_u = []
            r.readline()
            for line in r:
                mat_u.append([float(i) for i in line.rstrip().split()])
        mat_u = torch.FloatTensor(mat_u).transpose(0, 1)
        return torch.matmul(mat_u, mat_s) #.transpose(0, 1)

    def forward_nn(self, inputs, men_mask, ctx_mask, dist, gathers,dev=False):
        # Elmo contextualized embeddings
        #elmo_outputs = self.elmo(inputs)['elmo_representations'][0]

        #print("inputs:",inputs.shape)
        #print("elmo_outputs:",elmo_outputs.shape)
        #exit()
        #batch_size, _ = piece_idxs.size()
        #all_bert_outputs = self.bert(piece_idxs, attention_mask=attention_masks)
        #bert_outputs = all_bert_outputs[0]
        aetoken_len = inputs[1]
        inputs = inputs[0]

        if not dev:
            tt_token = inputs.cuda(device=2)
            all_bert_outputs = self.bert_model(tt_token)
        else:
            tt_token = inputs.cpu()
            self.bert_model.cpu()
            all_bert_outputs = self.bert_model(tt_token)
            self.bert_model.cuda(2)
        #print(all_bert_outputs)
        #hidden_states = all_bert_outputs[2]
        #print(len(all_bert_outputs[2]))
        #print(all_bert_outputs[0].shape,all_bert_outputs[1].shape)
        #print(hidden_states[0].shape)

        bert_outputs = all_bert_outputs[0]

        batch_size = bert_outputs.shape[0]

        idxs, masks, token_num, token_len = token_lens_to_idxs(aetoken_len)
        #print(token_num, token_len)
        bert_dim = bert_outputs.shape[-1]
        #batch_sizel = 1
        idxs = tt_token.new(idxs).unsqueeze(-1).expand(batch_size, -1, bert_dim) + 1
        masks = bert_outputs.new(masks).unsqueeze(-1)
        #print(tt_token.shape,masks.shape,idxs.shape,bert_outputs.shape)
        bert_outputs = torch.gather(bert_outputs, 1, idxs) * masks
        #bert_outputs = bert_outputs.view(batch_sizel, token_num, token_len, bert_dim)        
        bert_outputs = bert_outputs.view(batch_size, token_num, token_len, bert_dim)
        bert_outputs = bert_outputs.sum(2)
        #print(bert_outputs.shape,len(tokens))  
        bert_outputs =  bert_outputs.cuda()
        #print(bert_outputs.shape)
        #exit()

        elmo_outputs = bert_outputs

        #elmo_outputs = self.bert_model(inputs)['elmo_representations'][0]

        _, seq_len, feat_dim = elmo_outputs.size()
        gathers = gathers.unsqueeze(-1).unsqueeze(-1).expand(-1, seq_len, feat_dim)
        elmo_outputs = torch.gather(elmo_outputs, 0, gathers)

        men_attn = self.men_attn_linear_m(elmo_outputs).tanh()
        men_attn = self.men_attn_linear_o(men_attn)
        men_attn = men_attn + (1.0 - men_mask.unsqueeze(-1)) * -10000.0
        men_attn = men_attn.softmax(1)
        men_repr = (elmo_outputs * men_attn).sum(1)

        dist = self.dist_dropout(dist)
        ctx_attn = (self.ctx_attn_linear_c(elmo_outputs) +
                    self.ctx_attn_linear_m(men_repr.unsqueeze(1)) +
                    self.ctx_attn_linear_d(dist.unsqueeze(2))).tanh()
        ctx_attn = self.ctx_attn_linear_o(ctx_attn)

        ctx_attn = ctx_attn + (1.0 - ctx_mask.unsqueeze(-1)) * -10000.0
        ctx_attn = ctx_attn.softmax(1)
        ctx_repr = (elmo_outputs * ctx_attn).sum(1)

        # Classification
        final_repr = torch.cat([men_repr, ctx_repr], dim=1)
        final_repr = self.repr_dropout(final_repr)
        outputs = self.output_linear(final_repr)

        outputs_latent = None
        latent_label = self.feat_to_latent(final_repr) #.tanh()
        outputs_latent = self.latent_to_label(latent_label)
        outputs = outputs + self.latent_scalar * outputs_latent

        return outputs, outputs_latent

    def forward(self, inputs, labels, men_mask, ctx_mask, dist, gathers, inst_weights=None):
        outputs, outputs_latent = self.forward_nn(inputs, men_mask, ctx_mask, dist, gathers)
        loss = self.criterion(outputs, labels)
        return loss

    def _prediction(self, outputs, predict_top=True):
        _, highest = outputs.max(dim=1)
        highest = highest.int().tolist()
        preds = (outputs.sigmoid() > .5).int()
        if predict_top:
            for i, h in enumerate(highest):
                preds[i][h] = 1
        return preds

    def predict(self, inputs, men_mask, ctx_mask, dist, gathers, predict_top=True):
        self.eval()
        outputs, _ = self.forward_nn(inputs, men_mask, ctx_mask, dist, gathers)
        predictions = self._prediction(outputs, predict_top=predict_top)
        self.train()
        return predictions
