# from utils import *
import numpy as np
import re
import torch
from torch.autograd import Variable
from torch.utils.data import Dataset, DataLoader

def create_variable(tensor):
    # Do cuda() before wrapping with variable
    if torch.cuda.is_available():
        return Variable(tensor.cuda())
    else:
        return Variable(tensor)
def replace_halogen(string):
    """Regex to replace Br and Cl with single letters"""
    br = re.compile('Br')
    cl = re.compile('Cl')
    string = br.sub('R', string)
    string = cl.sub('L', string)
    return string
# Create necessary variables, lengths, and target
def make_variables(lines, properties,letters):
    sequence_and_length = [line2voc_arr(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences(vectorized_seqs, seq_lengths, properties)
def make_variables_seq(lines,letters):
    sequence_and_length = [line2voc_arr_seq(line,letters) for line in lines]
    vectorized_seqs = [sl[0] for sl in sequence_and_length]
    seq_lengths = torch.LongTensor([sl[1] for sl in sequence_and_length])
    return pad_sequences_seq(vectorized_seqs, seq_lengths)
def line2voc_arr(line,letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex(char,letters)) 
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex(unit,letters))
    return arr, len(arr)
def line2voc_arr_seq(line,letters):
    arr = []
    regex = '(\[[^\[\]]{1,10}\])'
    line = replace_halogen(line)
    char_list = re.split(regex, line)
    for li, char in enumerate(char_list):
        if char.startswith('['):
               arr.append(letterToIndex_seq(char,letters)) 
        else:
            chars = [unit for unit in char]

            for i, unit in enumerate(chars):
                arr.append(letterToIndex_seq(unit,letters))
    return arr, len(arr)
def letterToIndex(letter,smiles_letters):
    return smiles_letters.index(letter)
def letterToIndex_seq(letter,sequence_letters):
    return sequence_letters.index(letter)
# pad sequences and sort the tensor
def pad_sequences(vectorized_seqs, seq_lengths, properties):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
    seq_tensor = seq_tensor[perm_idx]

    # Also sort the target (countries) in the same order
    target = properties2tensor(properties)
    if len(properties):
        target = target[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), \
        create_variable(seq_lengths), \
        create_variable(target)
def pad_sequences_seq(vectorized_seqs, seq_lengths):
    seq_tensor = torch.zeros((len(vectorized_seqs), seq_lengths.max())).long()
    for idx, (seq, seq_len) in enumerate(zip(vectorized_seqs, seq_lengths)):
        seq_tensor[idx, :seq_len] = torch.LongTensor(seq)

    # Sort tensors by their length
    seq_lengths, perm_idx = seq_lengths.sort(0, descending=True)
#     print(seq_tensor)
    seq_tensor = seq_tensor[perm_idx]
    # Return variables
    # DataParallel requires everything to be a Variable
    return create_variable(seq_tensor), \
        create_variable(seq_lengths)

def construct_vocabulary(smiles_list,fname):
    """Returns all the characters present in a SMILES file.
       Uses regex to find characters/tokens of the format '[x]'."""
    add_chars = set()
    for i, smiles in enumerate(smiles_list):
        regex = '(\[[^\[\]]{1,10}\])'
        smiles = ds.replace_halogen(smiles)
        char_list = re.split(regex, smiles)
        for char in char_list:
            if char.startswith('['):
                add_chars.add(char)
            else:
                chars = [unit for unit in char]
                [add_chars.add(unit) for unit in chars]

    print("Number of characters: {}".format(len(add_chars)))
    with open(fname, 'w') as f:
        f.write('<pad>' + "\n")
        for char in add_chars:
            f.write(char + "\n")
    return add_chars
def properties2tensor(properties):
    property_ids = [train_dataset.get_property_id(
        property) for property in properties]
    return torch.DoubleTensor(property_ids)
def readLinesStrip(lines):
    for i in range(len(lines)):
        lines[i] = lines[i].rstrip('\n')
    return lines
def getProteinSeq(path,contactMapName):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    return seq
def getProtein(path,contactMapName,contactMap = True):
    proteins = open(path+"/"+contactMapName).readlines()
    proteins = readLinesStrip(proteins)
    seq = proteins[1]
    if(contactMap):
        contactMap = []
        for i in range(2,len(proteins)):
            contactMap.append(proteins[i])
        return seq,contactMap
    else:
        return seq

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


def getTrainDataSet(trainFoldPath):
    with open(trainFoldPath, 'r') as f:
        trainCpi_list = f.read().strip().split('\n')
    trainDataSet = [cpi.strip().split() for cpi in trainCpi_list]
    return trainDataSet#[[smiles, sequence, interaction],.....]
def getTestProteinList(testFoldPath):
    testProteinList = readLinesStrip(open(testFoldPath).readlines())[0].split()
    return testProteinList#['kpcb_2i0eA_full','fabp4_2nnqA_full',....]
def getSeqContactDict(contactPath,contactDictPath):# make a seq-contactMap dict 
    contactDict = open(contactDictPath).readlines()
    seqContactDict = {}
    for data in contactDict:
        _,contactMapName = data.strip().split(':')
        seq,contactMap = getProtein(contactPath,contactMapName)
        contactmap_np = [list(map(float, x.strip(' ').split(' '))) for x in contactMap]
        feature2D = np.expand_dims(contactmap_np, axis=0)
        feature2D = torch.FloatTensor(feature2D)    
        seqContactDict[seq] = feature2D
    return seqContactDict
def getLetters(path):
    with open(path, 'r') as f:
        chars = f.read().split()
    return chars
def getDataDict(testProteinList):
    dataDict = {}
    for x in testProteinList:#'xiap_2jk7A_full'
        xData = []
        protein = x.split('_')[0]
        print(protein)
        proteinActPath = ACTIVE_PATH+"/"+protein+"_actives_final.ism"
        proteinDecPath = DECOY_PATH+"/"+protein+"_decoys_final.ism"
        act = open(proteinActPath,'r').readlines()
        dec = open(proteinDecPath,'r').readlines()
        actives = [[x.split(' ')[0],1] for x in act] ######
        decoys = [[x.split(' ')[0],0] for x in dec]# test
        seq = getProtein(contactPath,x,contactMap = False)
        for i in range(len(actives)):
            xData.append([actives[i][0],seq,actives[i][1]])
        for i in range(len(decoys)):
            xData.append([decoys[i][0],seq,decoys[i][1]])
        print(len(xData))
        dataDict[x] = xData
    return dataDict
    
trainFoldPath = './data/DUDE/dataPre/DUDE-foldTrain1'
contactPath = './data/DUDE/contactMap'
contactDictPath = './data/DUDE/dataPre/DUDE-contactDict'
smileLettersPath  = './data/DUDE/voc/combinedVoc-wholeFour.voc'
seqLettersPath = './data/DUDE/voc/sequence.voc'
trainDataSet = getTrainDataSet(trainFoldPath)
seqContactDict = getSeqContactDict(contactPath,contactDictPath)
smiles_letters = getLetters(smileLettersPath)
sequence_letters = getLetters(seqLettersPath)

# testProteinList = getTestProteinList(testFoldPath)
testProteinList = ['kpcb_2i0eA_full','fabp4_2nnqA_full']
DECOY_PATH = './data/DUDE/decoy_smile'
ACTIVE_PATH = './data/DUDE/active_smile'
dataDict = getDataDict(testProteinList)

N_CHARS_SMI = len(smiles_letters)
N_CHARS_SEQ = len(sequence_letters)

# trainDataSet:[[smile,seq,label],....]    seqContactDict:{seq:contactMap,....}
train_dataset = ProDataset(dataSet = trainDataSet,seqContactDict = seqContactDict)
train_loader = DataLoader(dataset=train_dataset,
                          batch_size=1, shuffle=True,drop_last = True)