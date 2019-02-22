def check_verb_in_tags(word, morph):
    tags = set([x.tag.POS for x in morph.parse(word.lower()) if x.score > 0.1])
    if ('VERB' in tags) or ('PRTS' in tags) or ('INFN' in tags) or ('PRED' in tags) or ('GRND' in tags) or ('ADJS' in tags):
        if not ('PREP' in tags):
            return 'VERB'
    else:
        return morph.parse(word.lower())[0].tag.POS


def find_last_verb(sentence):
    words = sentence.split(' ')
    for word in words[-1::-1]:
        if check_verb_in_tags(word) == 'VERB':  # вообще, можно будет updat'ить для baselin'ов, но точно не сейчас
            return word
    return -1


def find_last_verb_position(sentence):
    word = find_last_verb(sentence)
    if word == -1:
        return -1, -1
    start = sentence.find(word)
    return start, start + len(word)


def join_words(words):
    res = ""
    for word in words[-1::-1]:
        if word.startswith('##'):
            res = word[2:] + res
        elif word.isalpha() or word.isdigit():
            return word + res
        else:
            return word


def check_word_pos(word, morph):
    tag = morph.parse(word.lower())[0].tag.POS
    return (tag not in ['CONJ', 'PRCL'])


def first_word(words):
    res = ""
    for word in words:
        if word.isalpha() or word.isdigit():
            if res == "":
                res = word
            else:
                return res
        elif word.startswith('##'):
            res = res + word[2:]
        else:
            return res
    return res
