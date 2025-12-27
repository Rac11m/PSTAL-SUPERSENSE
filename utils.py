import numpy as np
from tqdm import tqdm
from conllu import parse_incr
from collections import defaultdict


def get_sense_values():
    sense_values = set()
    for i, sent in enumerate(parse_incr(open("../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.full", encoding='UTF-8'))):
        for idx, tok in enumerate(sent):   
            sense_values.add(tok["frsemcor:noun"])
    return list(sense_values)

def get_sense_dict():
    sense_values = get_sense_values()
    label2id = {label: i for i, label in enumerate(sense_values)}
    id2label = {i: label for label, i in label2id.items()}
    return label2id, id2label 

def load_corpus(in_file):
    sents = parse_incr(open(in_file, encoding='UTF-8'))
    word_sent = {}
    upos_sent = {}
    for i, sent in enumerate(sents):
        word_sent[i] = []
        upos_sent[i] = []
        for token in sent:
            word_sent[i].append(token["form"])
            upos_sent[i].append(token["upos"]) 
        
    return word_sent, upos_sent

def token_alignment(input_sent, input_upos, tok_sent, emb_sent):
    '''
    input_sent: from conllu corpus, pair of (word,upos) 
    tok_sent: from the tokenizer of the pre-trained model
    '''
    # VALID_UPOS = [ADJ, ADP, ADV, AUX, CCONJ, DET, INTJ, NOUN, NUM, PART, PRON, PROPN, PUNCT, SCONJ, SYM, VERB, X]
    VALID_UPOS = ["NOUN", "NUM", "PROPN"]
    
    np_tok_sent = np.array(tok_sent.word_ids()) # convert the list of tokens to numpy array to use the np.where function
    tok_alignment = dict() # create a default dict contraining the ids corresponding to the word in tok_sent
    for id, w in enumerate(input_sent): 
        if (input_upos[id]).upper() in VALID_UPOS:
            tok_alignment[id] = (np.where(np_tok_sent == id)[0]).tolist()
        else:
            continue

    # get embedding of each word
    word_emb = dict.fromkeys(tok_alignment.keys(),None)
    for word, tok in tok_alignment.items():    
        word_emb[word] = emb_sent[tok].mean(dim=0)
    
    return tok_alignment, word_emb

def get_pairs(in_file, emb_alignment):
    pairs = []
    label2id, _ = get_sense_dict()
    with open(in_file, encoding="UTF-8") as f:
        for i, sent in tqdm(
            enumerate(parse_incr(f)),
            desc="Building (embedding, label) pairs"
        ):
            for idx, tok in enumerate(sent):
                if tok["upos"] in ["NOUN", "NUM", "PROPN"]:
                    pairs.append(
                        (emb_alignment[i][idx], label2id[tok["frsemcor:noun"]])
                    )
                    
    return pairs

def create_dataloader(dataset, batch_size, shuffle_mode):
    from torch.utils.data import DataLoader
    return DataLoader(dataset, batch_size=batch_size, shuffle=shuffle_mode)