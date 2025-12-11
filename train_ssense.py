import torch
from transformers import AutoTokenizer, AutoModel
from utils import load_corpus, token_alignment

in_file = "../PSTAL-MORPHtagging/pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.small"
model_url = "distilbert/distilbert-base-multilingual-cased" 

input_sent, upos_sent = load_corpus(in_file)
print(f"input_sent: {len(input_sent)}")

tokenize = AutoTokenizer.from_pretrained(model_url)
model = AutoModel.from_pretrained(model_url)

tok_alignment, emb_alignment = dict(), dict()

for idx, (sent, upos) in enumerate(zip(input_sent.items(), upos_sent.items())):
    tok_sent = tokenize(sent[1], is_split_into_words=True, return_tensors='pt')
    with torch.no_grad():    
        emb_sent = model(**tok_sent)['last_hidden_state'][0]
    tok_alignment[idx], emb_alignment[idx] = token_alignment(sent[1], upos[1], tok_sent, emb_sent)


print(f"emb_sent: {len(emb_alignment)}")