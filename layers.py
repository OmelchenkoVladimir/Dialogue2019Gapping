import pymorphy2
import torch
from extra import check_verb_in_tags


def top_layer(sent, tokenizer, model, sentence_tokenizer, morph, threshold=10):
    # на вход - предложение, токенизатор, модель, threshold, токенизатор для предложений (BasicTokenizer)
    sent = sent.replace('й', 'и').replace('ё', 'е')
    tok_sent = tokenizer.tokenize(sent)
    tok_verb_sent = sentence_tokenizer.tokenize(sent)
    verb_list = [word for word in tok_verb_sent if check_verb_in_tags(word, morph) == 'VERB']
    for verb in verb_list:
        verb_pos = sent.rfind(verb)
        tok_prefix = tokenizer.tokenize(sent[:(verb_pos+len(verb))])
        index = len(tok_prefix) + 1
        is_gapping, cV_pos, V_pos = middle_layer(tok_sent, verb, tokenizer, index, model, threshold, sent)
        if (is_gapping == 1):
            return is_gapping, cV_pos, V_pos
    return 0, "-1:-1", "-1:-1"


def middle_layer(tok_sent, verb, tokenizer, index, model, threshold, sent):
    # принимает на вход предложение, глагол, токенизатор, индекс, модель и threshold+sent; возвращает список форм глагола
    # на этом же слое должно происходить объединение результатов (есть ли гэппинг)
    # пока что всё делаем для одной формы глагола
    is_gapping = 0
    V_pos = []
    V_pos_res = ""
    cV_pos = "-1:-1"
    tok_verb = tokenizer.tokenize(verb)
    generated_words = bottom_layer(tok_sent, [tok_verb], tokenizer, index, model, threshold) # словарь
    for key in generated_words.keys():
        assert len(generated_words[key]) == len(tok_verb)
        assert len(generated_words[key]) > 0
        gapping_flag = 1
        for i in range(len(tok_verb)):
            if tok_verb[i] not in generated_words[key][i]:
                gapping_flag = 0
                break
        if (gapping_flag == 1):
            is_gapping = 1
            V_pos.append(key)
            if (cV_pos == "-1:-1"):
                tmp = sent.rfind(verb)
                cV_pos = f"{tmp}:{tmp + len(verb)}"
    for elem in V_pos: # TODO: реализовать быстрый (и правильный) поиск позиции слова
        if (elem != len(tok_sent)):
            tmp = sent.rfind(tok_sent[elem])
            V_pos_res += f"{tmp}:{tmp} "
    return is_gapping, cV_pos, V_pos_res


def bottom_layer(tok_sent, verb_list, tokenizer, index, model, threshold):
    # получаем предложение, список глаголов, токенизатор, индекс конца слова; возвращаем "похожие" токенизированные слова
    # также подаём на вход модель; threshold - количество получаемых элементов
    model.to('cuda')
    #tok_sent = tokenizer.tokenize(sent) # пока что
    res = {}
    if (len(verb_list) == 1):
        verb = verb_list[0] # список из k элементов (k - количество токенов)
        num_tokens = len(verb)
        for i in range(index+2, len(tok_sent)+1):
            if (i == len(tok_sent) or tok_sent[i].isalpha() or tok_sent[i].isdigit()):
                new_tok = tok_sent.copy()
                #print(tok_sent[i-1])
                #print('=====')
                # тогда перед этим всем вставляем num_tokens токенов
                for _ in range(num_tokens):
                    new_tok.insert(i, '[MASK]')
                indexed_tokens = torch.tensor([tokenizer.convert_tokens_to_ids(new_tok)]).to('cuda')
                segment_ids = torch.tensor([[0 for x in range(indexed_tokens.shape[1])]]).to('cuda')
                #print(indexed_tokens)
                #print(segment_ids)
                with torch.no_grad():
                    predictions = model(indexed_tokens, segment_ids)
                predictions = predictions.cpu()
                res[i] = []
                for j in range(i, i+num_tokens):
                    #predictions.cpu()[0, j].numpy().argsort(axis = 1).T[-10:][::-1].T
                    predicted_indexes = predictions[0, j].numpy().argsort()[-threshold:][::-1]
                    predicted_tokens = tokenizer.convert_ids_to_tokens(predicted_indexes)
                    res[i].append(predicted_tokens)
    return res


def pandas_top_layer(row, tokenizer, model, sentence_tokenizer, morph, threshold=10):
    # обёртка top layer'а для pandas
    print(row.name)
    sent = row['text']
    is_gapping, cV_pos, V_pos = top_layer(sent, tokenizer, model, sentence_tokenizer, morph, threshold)
    row['res_class'] = is_gapping
    row['res_cV'] = cV_pos
    row['res_V'] = V_pos
    return row
