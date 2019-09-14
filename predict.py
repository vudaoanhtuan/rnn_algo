import torch
import torch.nn as nn

def predict(model, inp, tokenizer, bos_token='^', eos_token='$', max_len=10):
    # inp: (1, seq_len)
    bos = tokenizer.word_index[bos_token]
    eos = tokenizer.word_index[eos_token]

    inp = tokenizer.texts_to_sequences([inp])
    inp = torch.tensor(inp)

    x, hs = model.encode(inp)
    x = torch.tensor([[bos]])
    output = []
    for i in range(max_len):
        x, hs = model.decode(x, hs) # x: (1, 1, vocab_size)
        x = x.argmax(dim=-1) # (1,1)
        if x.item() == eos:
            break
        output.append(x.item())
    s = tokenizer.sequences_to_texts([output])[0]
    return s

