import torch
import torch.nn as nn
from tqdm import tqdm
from model_ssense import SUPERSENSE_model
from transformers import AutoModel, AutoTokenizer
from utils import  create_dataloader, get_pairs, get_sense_values, load_corpus, token_alignment



def fit(model, epochs, train_loader, dev_loader):
  criterion = nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters()) 
  for epoch in range(epochs):
    model.train()
    total_loss = 0
    for (X, y) in tqdm(train_loader):
      optimizer.zero_grad()
      y_hat = model(X)  
      loss = criterion(y_hat, y)
      loss.backward()
      optimizer.step()
      total_loss += loss.item()  
    
    print(f"{epoch+1}/{epochs}")
    print("train_loss = {:.4f}".format(total_loss / len(train_loader.dataset)))
    print("dev_loss = {:.4f} dev_acc = {:.4f}".format(*perf(model, dev_loader, criterion)))

def perf(model, dev_loader, criterion):
  model.eval()
  total_loss = correct = total_tokens = 0
  for (X, y) in dev_loader:
    with torch.no_grad():
      y_hat = model(X) 
      total_loss += criterion(y_hat, y)
      y_pred = y_hat.argmax(dim=-1)  
      correct += (y_pred == y).sum().item()
      total_tokens += y.size(0)
      
  total = len(dev_loader.dataset)
  return total_loss / total, correct / total_tokens


def get_alignemnt_emb(pretrained_model, pretrained_tokenizer, input_sent, upos_sent):
    tok_alignment, emb_alignment = dict(), dict()
    
    for idx, (sent, upos) in tqdm(
        enumerate(zip(input_sent.items(), upos_sent.items())),
        total=len(input_sent),
        desc="Aligning & extracting embeddings"
    ):
        tok_sent = pretrained_tokenizer(sent[1], is_split_into_words=True, return_tensors='pt')
        with torch.no_grad():    
            emb_sent = pretrained_model(**tok_sent)['last_hidden_state'][0]
        tok_alignment[idx], emb_alignment[idx] = token_alignment(sent[1], upos[1], tok_sent, emb_sent)
    return tok_alignment, emb_alignment

if __name__ == "__main__":
    
    train_file = "../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.train"
    dev_file = "../pstal-etu/sequoia/sequoia-ud.parseme.frsemcor.simple.dev"
    model_url = "distilbert/distilbert-base-multilingual-cased" 
    batch_size = 32
    shuffle_mode = True
    sense_values = get_sense_values()

    # load pretrained model
    tokenize = AutoTokenizer.from_pretrained(model_url)
    pretrained_model = AutoModel.from_pretrained(model_url)
    print("Pretrained model loaded!")
    
    # load the train and dev corpus
    input_sent_train, upos_sent_train = load_corpus(train_file)
    input_sent_dev, upos_sent_dev = load_corpus(dev_file)
    print("Corpus loaded!")
    
    
    ## train and dev alignment's embeddings
    print("Trainset:")
    train_tok, train_emb = get_alignemnt_emb(pretrained_model, tokenize, input_sent_train, upos_sent_train)
    print("Devset:")
    dev_tok, dev_emb = get_alignemnt_emb(pretrained_model, tokenize, input_sent_dev, upos_sent_dev)
    print("Aligned embeddings extracted!")

    
    train_pairs = get_pairs(train_file, train_emb)
    dev_pairs = get_pairs(dev_file, dev_emb)
    train_loader = create_dataloader(train_pairs, batch_size, shuffle_mode)
    dev_loader = create_dataloader(dev_pairs, batch_size, shuffle_mode)
    print("Dataloaders created!")

    #### training
    print("Starting the training...!")
    
    output_size = len(sense_values) 
    emb_dim = pretrained_model.config.hidden_size
    hp = {
        "pretrained_model_type": pretrained_model.config.model_type,
        "model_type": "MLP_SUPERSENSE", 
        "embedding_dim": emb_dim, 
        "output_size": output_size
    }
  
    model = SUPERSENSE_model(emb_dim, output_size)
    fit(model=model, epochs=15, train_loader=train_loader, dev_loader=dev_loader)
  
    torch.save({"sense_tags": sense_values,
                "model_params": model.state_dict(),
                "hyperparams": hp}, f"supersense_{pretrained_model.config.model_type}.pt")        
