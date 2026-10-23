from nltk.corpus import wordnet as wn

def get_senses(word, pos=None):
    senses = wn.synsets(word.strip().lower(), pos=pos)
    result = []
    for s in senses:
        lemmas = s.lemmas()
        result.append(dict(synset=s.name(), pos=s.pos(), definition=s.definition(),
                           examples=s.examples(), synonyms=sorted(set(l.name().replace('_',' ') for l in lemmas)),
                           antonyms=sorted(set(a.name().replace('_',' ') for l in lemmas for a in l.antonyms())),
                           hypernyms=[x.name() for x in s.hypernyms()][:8],
                           hyponyms=[x.name() for x in s.hyponyms()][:8],
                           frequency=sum(l.count() for l in lemmas)))
    return result

def similarity_matrix(word, pos='n', limit=10):
    senses=wn.synsets(word, pos=pos)[:limit]
    return [s.name() for s in senses], [[a.wup_similarity(b) or 0 for b in senses] for a in senses]
