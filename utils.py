import numpy as np
from conllu import parse_incr
from collections import defaultdict

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
    tok_alignment = dict() # create en defaultdict contraining the ids corresponding to the word in tok_sent
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