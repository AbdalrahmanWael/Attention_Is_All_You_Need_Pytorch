
from tqdm.auto import tqdm
import os

import torch
from torch import nn
from torch.optim import Adam
from torch.utils.data import DataLoader

from transformer import Transformer
from data_loader import TranslationDataset
from utils.config import config

from tqdm.auto import tqdm

device = config['device']

from tqdm.auto import tqdm

def train_epoch(model, optimizer, loss_func, label_smoothing, train_loader, epoch_number, n_epochs, scheduler=None, device=device):
    model.train()
    train_loss = 0.0
    pbar = tqdm(total=len(train_loader), desc=f"Epoch {epoch_number + 1}/{n_epochs}: Training")
    for step, batch in enumerate(train_loader):
        src_tokens, trg_tokens = batch['src_tokens'].to(device), batch['trg_tokens'].to(device)
        src_mask, trg_mask = batch['src_mask'].to(device), batch['trg_mask'].to(device)

        # Log probabilities
        predictions = model(src_tokens, trg_tokens, src_mask, trg_mask).to(device)


        # print(' predictions.shape ', predictions.shape)
        # print(' predictions.dtype ', predictions.dtype)
            
        # print(' trg_tokens.shape ', trg_tokens.shape)
        # Reshape For the KL div Loss
        # (B, T, vocab_length) --> (B*T, vocab_length)
        predictions = predictions.view(-1, predictions.shape[-1])
            
        # (B, T, vocab_length) --> (B*T, 1)
        # trg_tokens = trg_tokens[:, 1:]
        trg_tokens = trg_tokens.reshape(-1, 1)


        # print(' predictions.shape ', predictions.shape)
        # print(' trg_tokens0.shape ', trg_tokens.shape)
            
            
        # Label smoothing
        # (B*T, 1) --> (B*T, vocab_length)
        smoothed_trg = label_smoothing(trg_tokens).to(device)
        # print(' smoothed_trg.shape ', smoothed_trg.shape)
        # print(' smoothed_trg.dtype ', smoothed_trg.dtype)
            
            
        optimizer.zero_grad()
        loss = loss_func(predictions, smoothed_trg)
        loss.backward()            
        optimizer.step()
            
        if scheduler is not None:
            scheduler.step()
                
        train_loss += loss.item() * src_tokens.size(0)

        current_lr = optimizer.param_groups[0]['lr']
        pbar.set_postfix(Loss=f"{loss.item():.4f}", LR=f"{current_lr:.6f}")
        pbar.update(1)

    pbar.close()

    train_loss /= len(train_loader.dataset)
    return train_loss

def validation(model, val_loader, loss_func, label_smoothing, epoch_number, n_epochs, device):
    model.eval()
    val_loss = 0.0
    pbar = tqdm(total=len(val_loader), desc=f"Epoch {epoch_number + 1}/{n_epochs}: Validation")
    with torch.no_grad():
        for step, batch in enumerate(val_loader):
            src_tokens, trg_tokens = batch['src_tokens'].to(device), batch['trg_tokens'].to(device)
            src_mask, trg_mask = batch['src_mask'].to(device), batch['trg_mask'].to(device)

            # Log probabilities
            predictions = model(src_tokens, trg_tokens, src_mask, trg_mask).to(device)

            # Reshape For the KL div Loss
            # (B, T, vocab_length) --> (B*T, vocab_length)
            predictions = predictions.view(-1, predictions.shape[-1])

            # (B, T, vocab_length) --> (B*T, 1)
            # trg_tokens = trg_tokens[:, 1:]
            trg_tokens = trg_tokens.reshape(-1, 1)

            # Label smoothing
            # (B*T, 1) --> (B*T, vocab_length)
            smoothed_trg = label_smoothing(trg_tokens).to(device)

            loss = loss_func(predictions, smoothed_trg)
            val_loss += loss.item() * src_tokens.size(0)

            pbar.set_postfix(Loss=f"{loss.item():.4f}")
            pbar.update(1)
    pbar.close()
    
    val_loss /= len(val_loader.dataset)
    return val_loss
    
def train(model, opt, loss_func, train_dataloader, valid_dataloader, label_smoothing, n_epochs, scheduler=None, run_name="Base_line_Model_Run1", device=device):
    best_val_loss = float("-inf")
    train_losses = []
    valid_losses = []

    for epoch in range(n_epochs):
        train_loss = train_epoch(model, opt, loss_func, label_smoothing, train_dataloader, epoch_number=epoch, n_epochs=n_epochs, scheduler=scheduler, device=device)
        train_losses.append(train_loss)

        valid_loss = validation(model, valid_dataloader, loss_func, label_smoothing, epoch_number=epoch, n_epochs=n_epochs, device=device)
        valid_losses.append(valid_loss)

        print(f"Epoch {epoch + 1}/{n_epochs}: Train Loss = {train_loss:.4f}, Val Loss = {valid_loss:.4f}")
            
        if valid_loss < best_val_loss:
            best_val_loss = valid_loss
            torch.save(model, f'{run_name}-epoch{epoch}-val_loss{valid_loss}')
            
        
    return train_losses, valid_losses

class CustomLRScheduler:
    def __init__(self, optimizer, d_model, n_warmup_steps):
        self.optimizer = optimizer
        self.d_model = d_model
        self.n_warmup_steps = n_warmup_steps
        self.current_step = 0

    def step(self):
        self.current_step += 1
        lr = self._get_learning_rate(self.current_step)
        self._set_learning_rate(lr)
        self.optimizer.step()

    def _get_learning_rate(self, step):
        warmup = self.n_warmup_steps
        return self.d_model ** (-0.5) * min(step ** (-0.5), step * warmup ** (-1.5))

    def _set_learning_rate(self, lr):
        for p in self.optimizer.param_groups:
            p['lr'] = lr

    def zero_grad(self):
        self.optimizer.zero_grad()


class LabelSmoothingDistribution(nn.Module):
    def __init__(self, smoothing_value: float, pad_token_id: int, trg_vocab_size: int, device: torch.device):
        assert 0.0 <= smoothing_value <= 1.0, "Smoothing value must be between 0.0 and 1.0"

        super(LabelSmoothingDistribution, self).__init__()

        self.confidence_value = 1.0 - smoothing_value
        self.smoothing_value = smoothing_value

        self.pad_token_id = pad_token_id
        self.trg_vocab_size = trg_vocab_size
        self.device = device

    def forward(self, trg_token_ids_batch: torch.Tensor) -> torch.Tensor:
        batch_size = trg_token_ids_batch.shape[0]
        smooth_target_distributions = torch.zeros((batch_size, self.trg_vocab_size), device=self.device)

        smooth_target_distributions.fill_(self.smoothing_value / (self.trg_vocab_size - 2))

        smooth_target_distributions.scatter_(1, trg_token_ids_batch, self.confidence_value)
        smooth_target_distributions[:, self.pad_token_id] = 0.

        smooth_target_distributions.masked_fill_(trg_token_ids_batch == self.pad_token_id, 0.)

        return smooth_target_distributions.to(torch.float32)
    

def _count_parameters(model):
    return sum(p.numel() for p in model.parameters() if p.requires_grad)

def _analyze_state_dict_shapes_and_names(model):
  """Prints detailed information about the model's state dictionary.

  This function iterates through the model's state dictionary and prints the names 
  and shapes of all parameters. It also raises an exception if any parameter is 
  not marked for training (requires_grad=False).

  """
  print("Keys of the model state_dict:")
  print(model.state_dict().keys())

  print("\nParameter details (name, shape):")
  for name, param in model.named_parameters():
    print(name, param.shape)
    if not param.requires_grad:
      raise Exception('Expected all parameters to be trainable. Found untrainable parameter:', name)
    

if __name__ == "__main__":
    from datasets import load_dataset, Dataset
    from data_loader import Tokenizer

    ds = load_dataset('iwslt2017', 'iwslt2017-de-en')
    tokenizer = Tokenizer(src_lang="de", trg_lang="en")

    batch_size = 128 # config['HyperParemeters_Base_recommended']['batch_size']
    warmup = 1000 #config['HyperParemeters_Base_recommended']['warmup']
    n_epochs = 10

    n_layers = config['BASELINE_MODEL_CONFIG']['num_layers']
    d_model = config['BASELINE_MODEL_CONFIG']['d_model']
    n_heads = config['BASELINE_MODEL_CONFIG']['num_heads']
    dropout_p = config['BASELINE_MODEL_CONFIG']['dropout_prob']
    max_length = 50 #config['BASELINE_MODEL_CONFIG']['max_length']

    label_smoothing_value = config['BASELINE_MODEL_CONFIG']['label_smoothing_value']

    training_dataset = TranslationDataset(ds['train']['translation'], tokenizer, max_length=max_length, batch_size=batch_size, vocab_min_freq=3)

    trg_vocab = training_dataset.trg_vocab
    src_vocab = training_dataset.src_vocab

    validation_dataset = TranslationDataset(ds['validation']['translation'], tokenizer, src_vocab=src_vocab, trg_vocab=trg_vocab, 
                                        max_length=max_length, batch_size=batch_size)

    pad_token_id = training_dataset.src_vocab.word2idx[config['SpecialTokens']['PAD_TOKEN']]

    train_dataloader = DataLoader(training_dataset, batch_size=batch_size, shuffle=True)
    valid_dataloader = DataLoader(validation_dataset, batch_size=batch_size, shuffle=False)


    src_vocab_size = len(training_dataset.src_vocab)
    trg_vocab_size = len(training_dataset.trg_vocab)

    

    model = Transformer(
    src_vocab_length=src_vocab_size,
    trg_vocab_length=trg_vocab_size,
    max_length= max_length,
    d_model=d_model,
    n_heads=n_heads,
    n_layers=n_layers,
    dropout_p=dropout_p
    ).to(device)

    _count_parameters(model)

    from torch.optim import Adam
    pad_token_id = training_dataset.src_vocab.word2idx[config['SpecialTokens']['PAD_TOKEN']]

    kl_div_loss = nn.KLDivLoss(reduction="batchmean")
    label_smoothing = LabelSmoothingDistribution(label_smoothing_value, pad_token_id, trg_vocab_size, device)

    optimizer = Adam(model.parameters(), lr=1e-4, betas=(0.9, 0.98), eps=1e-9)
    scheduler = CustomLRScheduler(optimizer, d_model, 100)

    log_dir = 'logs'
    os.makedirs(log_dir, exist_ok=True)
    n_epochs = 10   
    train_losses, val_losses = train(
        model=model,
        opt=optimizer,
        loss_func=kl_div_loss,
        train_dataloader=train_dataloader,
        valid_dataloader=valid_dataloader,
        label_smoothing=label_smoothing,
        scheduler=scheduler,
        n_epochs=n_epochs,
        run_name="Base_line_Model_Run1",
        device=device
    )

    torch.save(model, 'logs/maaan.pt')

    from translation_decoder import TranslationDecoder
    translation_decoder = TranslationDecoder(model, src_vocab=training_dataset.src_vocab,
                                         trg_vocab=training_dataset.trg_vocab, device=device, max_length=max_length)
    
    def detokenize(b_tokens, idx2word):
        output_sentences = []
        for tokens in b_tokens:
            sentence = []
            for token in tokens:
                word = idx2word[token.item()]
                if word == config['SpecialTokens']['EOS_TOKEN']:
                    break
                sentence.append(word)
            output_sentences.append(' '.join(sentence))
        return output_sentences

    inp = next(iter(valid_dataloader))
    ex_src_tokens = inp['src_tokens']
    ex_trg_tokens = inp['trg_tokens']
    ex_src_mask = inp['src_mask']
    ex_trg_mask = inp['trg_mask']
    print('output_text')
    output_texts = translation_decoder.greedy_decoding(ex_src_tokens.to(device), ex_src_mask.to(device))
    print(output_texts)