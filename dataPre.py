from model import *
from utils import *
from torch.utils.data import Dataset, DataLoader

class ProDataset(Dataset):
    # Initialize your data, download, etc.
    def __init__(self,dataSet,seqContactDict):
        self.dataSet = dataSet#list:[[smile,seq,label],....]
        self.len = len(dataSet)
        self.dict = seqContactDict#dict:{seq:contactMap,....}
        self.properties = [int(x[2]) for x in dataSet]# labels
        self.property_list = list(sorted(set(self.properties)))

    def __getitem__(self, index):
        smiles,seq,label = self.dataSet[index]
        contactMap = self.dict[seq]
        return smiles, contactMap, int(label)

    def __len__(self):
        return self.len

    def get_properties(self):
        return self.property_list

    def get_property(self, id):
        return self.property_list[id]

    def get_property_id(self, property):
        return self.property_list.index(property)
    
testFoldPath = './data/DUDE/dataPre/DUDE-foldTest3'
trainFoldPath = './data/DUDE/dataPre/DUDE-foldTrain3'
contactPath = './data/DUDE/contactMap'
contactDictPath = './data/DUDE/dataPre/DUDE-contactDict'
smileLettersPath  = './data/DUDE/voc/combinedVoc-wholeFour.voc'
seqLettersPath = './data/DUDE/voc/sequence.voc'
print('get train datas....')
trainDataSet = getTrainDataSet(trainFoldPath)
print('get seq-contact dict....')
seqContactDict = getSeqContactDict(contactPath,contactDictPath)
print('get letters....')
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)

# testProteinList = getTestProteinList(testFoldPath)# whole foldTest
# testProteinList = ['kpcb_2i0eA_full']# a protein of fold1Test
testProteinList = ['tryb1_2zebA_full','mcr_2oaxE_full', 'cxcr4_3oduA_full']# protein of fold3Test
DECOY_PATH = './data/DUDE/decoy_smile'
ACTIVE_PATH = './data/DUDE/active_smile'
print('get protein-seq dict....')
dataDict = getDataDict(testProteinList,ACTIVE_PATH,DECOY_PATH,contactPath)

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

print('train loader....')
# trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
train_dataset = ProDataset(dataSet = trainDataSet,seqContactDict = seqContactDict)
train_loader = DataLoader(dataset = train_dataset,batch_size=1, shuffle=True,drop_last = True)

print('model args...')

modelArgs = {}
modelArgs['batch_size'] = 1
modelArgs['lstm_hid_dim'] = 64
modelArgs['d_a'] = 32
modelArgs['r'] = 10
modelArgs['n_chars_smi'] = 247
modelArgs['n_chars_seq'] = 21
modelArgs['dropout'] = 0.2
modelArgs['in_channels'] = 8
modelArgs['cnn_channels'] = 32
modelArgs['cnn_layers'] = 4
modelArgs['emb_dim'] = 30
modelArgs['dense_hid'] = 64
modelArgs['task_type'] = 0
modelArgs['n_classes'] = 1

print('train args...')

trainArgs = {}
trainArgs['model'] = DrugVQA(modelArgs,block = ResidualBlock).cuda()
trainArgs['epochs'] = 30
trainArgs['lr'] = 0.0007
trainArgs['train_loader'] = train_loader
trainArgs['doTest'] = True
trainArgs['test_proteins'] = testProteinList
trainArgs['testDataDict'] = dataDict
trainArgs['seqContactDict'] = seqContactDict
trainArgs['use_regularizer'] = False
trainArgs['penal_coeff'] = 0.03
trainArgs['clip'] = True
trainArgs['criterion'] = torch.nn.BCELoss()
trainArgs['optimizer'] = torch.optim.Adam(trainArgs['model'].parameters(),lr=trainArgs['lr'])
trainArgs['doSave'] = True
trainArgs['saveNamePre'] = 'DUDE30Res-fold3-'

print('train args over...')
