from googletrans import Translator
import torch
import transformers
import itertools
from transformers import AutoTokenizer, AutoModelForMaskedLM, AutoModel

def translate_sentence(text):
    translator = Translator()
    return translator.translate(text, src='en', dest='tr').text

def allign_texts(src, tgt):


    # model = AutoModel.from_pretrained('FacebookAI/xlm-roberta-large')
    # tokenizer = AutoTokenizer.from_pretrained('FacebookAI/xlm-roberta-large')

    model = transformers.BertModel.from_pretrained('bert-base-multilingual-cased')
    tokenizer = transformers.BertTokenizer.from_pretrained('bert-base-multilingual-cased')

    # pre-processing
    sent_src, sent_tgt = src.strip().split(), tgt.strip().split()
    token_src, token_tgt = [tokenizer.tokenize(word) for word in sent_src], [tokenizer.tokenize(word) for word in
                                                                             sent_tgt]
    wid_src, wid_tgt = [tokenizer.convert_tokens_to_ids(x) for x in token_src], [tokenizer.convert_tokens_to_ids(x) for
                                                                                 x in token_tgt]
    ids_src, ids_tgt = tokenizer.prepare_for_model(list(itertools.chain(*wid_src)), return_tensors='pt',
                                                   model_max_length=tokenizer.model_max_length, truncation=True)[
        'input_ids'], tokenizer.prepare_for_model(list(itertools.chain(*wid_tgt)), return_tensors='pt', truncation=True,
                                                  model_max_length=tokenizer.model_max_length)['input_ids']
    sub2word_map_src = []
    for i, word_list in enumerate(token_src):
        sub2word_map_src += [i for x in word_list]
    sub2word_map_tgt = []
    for i, word_list in enumerate(token_tgt):
        sub2word_map_tgt += [i for x in word_list]

    # alignment
    align_layer = 8
    threshold = 1e-3
    model.eval()
    with torch.no_grad():
        out_src = model(ids_src.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]
        out_tgt = model(ids_tgt.unsqueeze(0), output_hidden_states=True)[2][align_layer][0, 1:-1]

        dot_prod = torch.matmul(out_src, out_tgt.transpose(-1, -2))

        softmax_srctgt = torch.nn.Softmax(dim=-1)(dot_prod)
        softmax_tgtsrc = torch.nn.Softmax(dim=-2)(dot_prod)

        softmax_inter = (softmax_srctgt > threshold) * (softmax_tgtsrc > threshold)

    align_subwords = torch.nonzero(softmax_inter, as_tuple=False)
    align_words = set()
    for i, j in align_subwords:
        align_words.add((sub2word_map_src[i], sub2word_map_tgt[j]))

    # printing
    class color:
        PURPLE = '\033[95m'
        CYAN = '\033[96m'
        DARKCYAN = '\033[36m'
        BLUE = '\033[94m'
        GREEN = '\033[92m'
        YELLOW = '\033[93m'
        RED = '\033[91m'
        BOLD = '\033[1m'
        UNDERLINE = '\033[4m'
        END = '\033[0m'

    res_dict = dict()
    for i, j in sorted(align_words):
        res_dict[sent_src[i]]=sent_tgt[j]
        print(f'{color.BOLD}{color.BLUE}{sent_src[i]}{color.END}==={color.BOLD}{color.RED}{sent_tgt[j]}{color.END}')
    return res_dict

if __name__ == '__main__':
    src = """USS Enterprise CVN-65 (CVN-65) and UK HMS Victory CVN-65 joined 1 June 2022, USS Missouri (BB-63) arrived at Pearl Harbor 31 January 2024, also by Naida Hakirevic Prevljak . On the day of the launch of HMS Victory (H41), HMS Queen Elizabeth actively participated. Admiral John Doe was overseeing operations."""
    tgt = translate_sentence(text=src)
    res = allign_texts(src, tgt)
    print("finish")