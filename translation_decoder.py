import torch
from torch import nn

from utils.config import config

class TranslationDecoder:
    def __init__(self, model, src_vocab, trg_vocab, device, max_length=50):
        self.model = model 
        self.src_vocab = src_vocab    
        self.trg_vocab = trg_vocab
        self.device = device
        self.max_length = max_length 

    def get_src_mask(self, src_tokens, pad_token):

        batch_size = src_tokens.size(0)
        # masks (B, 1, 1, T)
        src_mask = (src_tokens != pad_token).view(batch_size, 1, 1, -1)

        return src_mask
    
    def get_trg_mask(self, trg_tokens, pad_token):
        dev = trg_tokens.device
        batch_size = trg_tokens.size(0)
        sequence_length = trg_tokens.size(-1)


        trg_padding_mask = (trg_tokens != pad_token).view(batch_size, 1, 1, -1)
        trg_tri_mask = torch.triu(torch.ones((batch_size, 1, sequence_length, sequence_length), device=dev) == 1)
        trg_tri_mask = trg_tri_mask.transpose(2, 3)
        trg_mask = trg_padding_mask & trg_tri_mask # (B, 1, T, T)

        return trg_mask # masks (B, 1, 1, T)
        
    
    def greedy_decoding(self, src_tokens, src_mask):
        self.model.eval()
        dev = src_tokens.device 
        pad_token = self.trg_vocab.word2idx[config['SpecialTokens']['PAD_TOKEN']]
        sos_token = self.trg_vocab.word2idx[config['SpecialTokens']['BOS_TOKEN']]
        eos_token = self.trg_vocab.word2idx[config['SpecialTokens']['EOS_TOKEN']]

        # Initialize with SOS tokens (B, 1)
        trg_tokens = torch.full((src_tokens.size(0), 1), sos_token, dtype=torch.long, device=dev)
        # trg_tokens shape: (batch_size, 1)

        with torch.no_grad():
            # (batch_size, sequence_length, d_model)
            memory = self.model.encode(src_tokens, src_mask)
            
            for _ in range(self.max_length):
                # (batch_size, 1, current_sequence_length, current_sequence_length)
                trg_mask = self.get_trg_mask(trg_tokens, pad_token)
                # (batch_size, current_sequence_length, trg_vocab_size)
                output = self.model.decode(trg_tokens, memory, trg_mask, src_mask)
                # output[:, -1] shape: (batch_size, trg_vocab_size)
                # output_token shape: (batch_size, 1)
                output_token = output[:, -1].argmax(dim=-1, keepdim=True)
                
                # (batch_size, current_sequence_length + 1)
                trg_tokens = torch.cat([trg_tokens, output_token], dim=1)
                
                if (output_token == eos_token).all():
                    break
            
            output_sentences = []
            for tokens in trg_tokens:
                sentence = []
                for token in tokens:
                    word = self.trg_vocab.idx2word[token.item()]
                    if word == config['SpecialTokens']['EOS_TOKEN']:
                        break
                    sentence.append(word)
                output_sentences.append(' '.join(sentence))
                # Each sentence is a string of words joined by spaces
            
        return output_sentences