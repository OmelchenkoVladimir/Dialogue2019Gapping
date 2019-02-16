import pymorphy2


def check_verb_in_tags(word):
    morph = pymorphy2.MorphAnalyzer()
    tags = set([x.tag.POS for x in morph.parse(word.lower())])
    if ('VERB' in tags) or ('PRTS' in tags) or ('INFN' in tags) or ('PRED' in tags) or ('GRND' in tags) or ('ADJS' in tags):
        return 'VERB'
    else:
        return morph.parse(word.lower())[0].tag.POS


def find_last_verb(sentence):
    words = sentence.split(' ')
    for word in words[-1::-1]:
        if check_verb_in_tags(word) == 'VERB':
            return word
    return -1


def find_last_verb_position(sentence):
    word = find_last_verb(sentence)
    if word == -1:
        return -1, -1
    start = sentence.find(word)
    return start, start + len(word)
