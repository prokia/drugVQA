import torch.nn as nn
import torch
from torch.autograd import Variable
import torch.nn.functional as F

def conv3x3(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=3,
                     stride=stride, padding=1, bias=False)
def conv5x5(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=5,
                     stride=stride, padding=2, bias=False)
def conv1x1(in_channels, out_channels, stride=1):
    return nn.Conv2d(in_channels, out_channels, kernel_size=1,
                     stride=stride, padding=0, bias=False)
# Residual block
class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, stride=1, downsample=None):
        super(ResidualBlock, self).__init__()
        self.conv1 = conv5x5(in_channels, out_channels, stride)
        self.bn1 = nn.BatchNorm2d(out_channels)
        self.elu = nn.ELU(inplace=True)
        self.conv2 = conv3x3(out_channels, out_channels)
        self.bn2 = nn.BatchNorm2d(out_channels)
        self.downsample = downsample

    def forward(self, x):
        residual = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.elu(out)
        out = self.conv2(out)
        out = self.bn2(out)
        if self.downsample:
            residual = self.downsample(x)
        out += residual
        out = self.elu(out)
        return out

class DrugVQA(torch.nn.Module):
    """
    The class is an implementation of the DrugVQA model including regularization and without pruning. 
    Slight modifications have been done for speedup
    
    """
    def __init__(self,batch_size,lstm_hid_dim,d_a,r,block,n_chars_smi,n_chars_seq, emb_dim=30,vocab_size=None,num_classes=10,type=0,n_classes = 1):
        """
        Initializes parameters suggested in paper
 
        Args:
            batch_size  : {int} batch_size used for training
            lstm_hid_dim: {int} hidden dimension for lstm
            d_a         : {int} hidden dimension for the dense layer
            r           : {int} attention-hops or attention heads
            emb_dim     : {int} embeddings dimension
            vocab_size  : {int} size of the vocabulary
            type        : [0,1] 0-->binary_classification 1-->multiclass classification
            n_classes   : {int} number of classes
 
        Returns:
            self
        """
        super(DrugVQA,self).__init__()
        #rnn
        self.embeddings = nn.Embedding(n_chars_smi, emb_dim)
        self.seq_embed = nn.Embedding(n_chars_seq,emb_dim)
        self.lstm = torch.nn.LSTM(emb_dim,lstm_hid_dim,2,batch_first=True,bidirectional=True,dropout=0.2) 
        self.linear_first = torch.nn.Linear(2*lstm_hid_dim,d_a)
        self.linear_first.bias.data.fill_(0)
        self.linear_second = torch.nn.Linear(d_a,r)
        self.linear_second.bias.data.fill_(0)
        self.linear_first_seq = torch.nn.Linear(32,d_a)
        self.linear_first_seq.bias.data.fill_(0)
        self.linear_second_seq = torch.nn.Linear(d_a,r)
        self.linear_second_seq.bias.data.fill_(0)
        #cnn
        self.in_channels = 8
        self.conv = conv3x3(1, 8)
        self.bn = nn.BatchNorm2d(8)
        self.elu = nn.ELU(inplace=False)
        self.layer1 = self.make_layer(block, 16, 5)
        self.layer2 = self.make_layer(block, 32, 5)
        self.layer3 = self.make_layer(block, 32, 5)
        
        self.n_classes = n_classes
        self.linear_final_step = torch.nn.Linear(lstm_hid_dim*2+d_a,100)
        self.linear_final = torch.nn.Linear(100,self.n_classes)
        self.batch_size = batch_size
        self.lstm_hid_dim = lstm_hid_dim
        self.hidden_state = self.init_hidden()
        self.seq_hidden_state = self.init_hidden()
        self.r = r
        self.type = type
        
    def softmax(self,input, axis=1):
        """
        Softmax applied to axis=n
 
        Args:
           input: {Tensor,Variable} input on which softmax is to be applied
           axis : {int} axis on which softmax is to be applied
 
        Returns:
            softmaxed tensors
        """
        input_size = input.size()
        trans_input = input.transpose(axis, len(input_size)-1)
        trans_size = trans_input.size()
        input_2d = trans_input.contiguous().view(-1, trans_size[-1])
        soft_max_2d = F.softmax(input_2d)
        soft_max_nd = soft_max_2d.view(*trans_size)
        return soft_max_nd.transpose(axis, len(input_size)-1)
       
        
    def init_hidden(self):
        return (Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim).cuda()),Variable(torch.zeros(4,self.batch_size,self.lstm_hid_dim)).cuda())
    
    def make_layer(self, block, out_channels, blocks, stride=1):
        downsample = None
        if (stride != 1) or (self.in_channels != out_channels):
            downsample = nn.Sequential(
                conv3x3(self.in_channels, out_channels, stride=stride),
                nn.BatchNorm2d(out_channels))
        layers = []
        layers.append(block(self.in_channels, out_channels, stride, downsample))
        self.in_channels = out_channels
        for i in range(1, blocks):
            layers.append(block(out_channels, out_channels))
        return nn.Sequential(*layers)
        
    # x1 = smiles , x2 = contactMap
    def forward(self,x1,x2):
        embeddings = self.embeddings(x1)         
        outputs, self.hidden_state = self.lstm(embeddings,self.hidden_state)    
        x1 = F.tanh(self.linear_first(outputs))       
        x1 = self.linear_second(x1)       
        x1 = self.softmax(x1,1)       
        attention = x1.transpose(1,2)        
        sentence_embeddings = attention@outputs       
        avg_sentence_embeddings = torch.sum(sentence_embeddings,1)/self.r  #multi head
         
        out = self.conv(x2)
        out1 = self.bn(out)
        out2 = self.elu(out1)
        out3 = self.layer1(out2)
        out3 = self.layer2(out3)
        out3 = self.layer3(out3)
        o1 = torch.mean(out3,2)
        o1 = o1.permute(0,2,1)
        x = F.tanh(self.linear_first_seq(o1))       
        x = self.linear_second_seq(x)       
        x = self.softmax(x,1)       
        seq_attention = x.transpose(1,2)
        seq_embeddings = seq_attention@o1      
        avg_seq_embeddings = torch.sum(seq_embeddings,1)/self.r
        
        sscomplex = torch.cat([avg_sentence_embeddings,avg_seq_embeddings],dim=1) 
        sscomplex = F.relu(self.linear_final_step(sscomplex))
        self.seq_hidden_state =  self.init_hidden()  
        if not bool(self.type):
            output = F.sigmoid(self.linear_final(sscomplex))
            return output,attention
        else:
            return F.log_softmax(self.linear_final(sscomplex)),attention