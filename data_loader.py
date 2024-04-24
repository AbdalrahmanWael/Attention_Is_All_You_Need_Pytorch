
import spacy
import torch
import os
from utils.config import config

class Vocabulary:
    def __init__(self, min_freq=2):
        self.PAD = config['SpecialTokens']['PAD_TOKEN']
        self.BOS = config['SpecialTokens']['BOS_TOKEN']
        self.EOS = config['SpecialTokens']['EOS_TOKEN']
        self.UNK = config['SpecialTokens']['UNK_TOKEN']
        
        self.word2idx = {self.PAD: 0, self.BOS: 1, self.EOS: 2, self.UNK:3}
        self.idx2word = {v: k for k, v in self.word2idx.items()}
        self.min_freq = min_freq
        self.max_len = -1
    
    def add_word(self, word):
        if word not in self.word2idx:
            self.word2idx[word] = len(self.word2idx)
            self.idx2word[len(self.idx2word)] = word    
            if len(word) > self.max_len: self.max_len = len(word)
    
    def create_vocab(self, texts):
        counter = {}
        for text in texts:
            for word in text:
                counter[word] = counter.get(word, 0) + 1
        
        for word, freq in counter.items():
            if freq >= self.min_freq and word not in self.word2idx:
                self.add_word(word)
    
    def numerize(self, text):
        return [self.word2idx.get(word, self.word2idx[self.UNK]) for word in text]

    def __len__(self):
        return len(self.word2idx.keys())
    


class Tokenizer:
    def __init__(self, src_lang="de", trg_lang="en", custom_tokenizer=None):
        self.src_lang = src_lang
        self.trg_lang = trg_lang
        self.custom_tokenizer = custom_tokenizer
        
        if custom_tokenizer is not None:
            if not callable(custom_tokenizer):
                raise ValueError("Custom tokenizer must be a callable")
        
        if custom_tokenizer is None:
            self.spacy_src = self._load_spacy_model(src_lang)
            self.spacy_trg = self._load_spacy_model(trg_lang)
    
    def _load_spacy_model(self, lang):
        lang_to_model = config['lang_to_model']
        
        if lang not in lang_to_model:
            raise ValueError(f"Unsupported language: {lang}")
        
        model_name = lang_to_model[lang]
        try:
            return spacy.load(model_name)
        except OSError:
            print(f"Downloading SpaCy model for {lang}...")
            try:
                os.system(f"python -m spacy download {model_name}")
                return spacy.load(model_name)
            except OSError as e:
                raise RuntimeError(f"Failed to download SpaCy model for {lang}: {e}")
    
    def tokenize_src(self, text):
        if self.custom_tokenizer is not None:
            return self.custom_tokenizer(text)
        return self.tokenize(text, self.spacy_src)
    
    def tokenize_trg(self, text):
        if self.custom_tokenizer is not None:
            return self.custom_tokenizer(text)
        return self.tokenize(text, self.spacy_trg)
    
    def tokenize(self, text, tokenizer):
        return [tok.text for tok in tokenizer.tokenizer(text)]
    
    def detokenize(self, tokens):
        return " ".join(tokens)
    
    def __call__(self, src_text, trg_text):
        src_tokens = self.tokenize_src(src_text)
        trg_tokens = self.tokenize_trg(trg_text)
        return src_tokens, trg_tokens




class TranslationDataset:
    def __init__(self, dataset, tokenizer, ext=('de', 'en'), batch_size=32, train=True):
        self.dataset = dataset
        self.tokenizer = tokenizer
        self.ext = ext
        self.batch_size = batch_size
        
        # Create Vocabulary
        self.src_vocab, self.trg_vocab = self.get_vocabs()


    def get_vocabs(self):
        src_words = []
        trg_words = []
    
        for sample in self.dataset:
            src_sentence = sample[self.ext[0]]
            trg_sentence = sample[self.ext[1]]
    
            # Tokenize
            src_sentence_tokens, trg_sentence_tokens = self.tokenizer(src_sentence, trg_sentence)
            src_words.append(src_sentence_tokens)
            trg_words.append(trg_sentence_tokens)
    
        # Create Vocabulary
        src_vocab = Vocabulary(min_freq=2)
        trg_vocab = Vocabulary(min_freq=2)
        src_vocab.create_vocab(src_words)
        trg_vocab.create_vocab(trg_words)
    
        return src_vocab, trg_vocab
    
    def get_src_mask(self, src_tokens, pad_token):
        
        batch_size = src_tokens.size(0)
        # masks (B, 1, 1, T)
        src_mask = (src_tokens != pad_token).view(batch_size, 1, 1, -1)

        return src_mask
    
    def get_trg_mask(self, trg_tokens, pad_token):

        batch_size = trg_tokens.size(0)
        sequence_length = trg_tokens.size(-1)
        

        trg_padding_mask = (trg_tokens != pad_token).view(batch_size, 1, 1, -1)
        trg_tri_mask = torch.triu(torch.ones(batch_size, 1, sequence_length, sequence_length) == 1)
        trg_tri_mask = trg_tri_mask.transpose(2, 3)
        trg_mask = trg_padding_mask & trg_tri_mask # (B, 1, T, T)

        return trg_mask # masks (B, 1, 1, T)

    def get_masks(self, src_tokens, trg_tokens, pad_token):
        src_mask = self.get_src_mask(src_tokens, pad_token)
        trg_mask = self.get_trg_mask(trg_tokens, pad_token)

        return src_mask, trg_mask

    def preprocess(self, example):
        src_text = example[self.ext[0]]
        trg_text = example[self.ext[1]]

        src_tokens = self.tokenizer.tokenize_src(src_text)
        trg_tokens = self.tokenizer.tokenize_trg(trg_text)

        # Numerize the text
        src_tokens = self.src_vocab.numerize(src_tokens)
        trg_tokens = self.trg_vocab.numerize(trg_tokens)

        # Truncate and pad the tokens to the maximum_length
        src_max_len = self.src_vocab.max_len
        trg_max_len = self.trg_vocab.max_len
        
        
        src_tokens = src_tokens[:src_max_len]
        trg_tokens = trg_tokens[:trg_max_len]
        
        PAD = config['SpecialTokens']['PAD_TOKEN']
        src_tokens += [self.src_vocab.word2idx[PAD]] * ( src_max_len - len(src_tokens))
        trg_tokens += [self.trg_vocab.word2idx[PAD]] * ( trg_max_len - len(trg_tokens))

        # Add BOS and EOS tokens to the target tokens
        BOS = config['SpecialTokens']['BOS_TOKEN']
        EOS = config['SpecialTokens']['EOS_TOKEN']
        trg_tokens = [self.trg_vocab.word2idx[BOS]] + trg_tokens + [self.trg_vocab.word2idx[EOS]]

        src_tokens = torch.tensor(src_tokens).view(1, -1)
        trg_tokens = torch.tensor(trg_tokens).view(1, -1)
        
        # Create Attention Masks
        src_mask, trg_mask = self.get_masks(src_tokens, trg_tokens, pad_token=0)
        
        # Shift Target Tokens by 1 
        trg_tokens = trg_tokens[:, :-1] # (B, T)
        
        return {
            "src_tokens": src_tokens,
            "trg_tokens": trg_tokens,
            "src_mask": src_mask,
            "trg_mask": trg_mask
        }
    def __len__(self):
        return len(self.dataset)
        
    def __getitem__(self, index):
        return self.preprocess(self.dataset[index])                          
    




if __name__ == "__main__":
    from datasets import load_dataset, Dataset
    
    # Load the dataset
    ds = load_dataset('iwslt2017', 'iwslt2017-de-en')
    
    # Create an instance of the Tokenizer
    tokenizer = Tokenizer(src_lang="de", trg_lang="en")
    
    # Create an instance of the TranslationDataset
    batch_size = 4
    dataset = TranslationDataset(ds['train']['translation'], tokenizer, batch_size=batch_size)
    
    # Get a single example from the dataset
    example = dataset[0]
    
    # Check the shapes and contents of the example
    print("Source tokens shape:", example['src_tokens'].shape)
    print("Target tokens shape:", example['trg_tokens'].shape)
    print("Source mask shape:", example['src_mask'].shape)
    print("Target mask shape:", example['trg_mask'].shape)
    
    print("\nSource tokens:")
    print(example['src_tokens'])
    print("\nTarget tokens:")
    print(example['trg_tokens'])
    print("\nSource mask:")
    print(example['src_mask'])
    print("\nTarget mask:")
    print(example['trg_mask'])
        
    from torch.utils.data import DataLoader

    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

    for batch in dataloader:
        src_tokens = batch['src_tokens']
        trg_tokens = batch['trg_tokens']
        src_mask = batch['src_mask']
        trg_mask = batch['trg_mask']
        
        print("\nBatch:")
        print("Source tokens shape:", src_tokens.shape)
        print("Target tokens shape:", trg_tokens.shape)
        print("Source mask shape:", src_mask.shape)
        print("Target mask shape:", trg_mask.shape)

        break
