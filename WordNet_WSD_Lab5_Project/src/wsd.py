import math
import re
from collections import Counter
from nltk.corpus import wordnet as wn
from nltk.wsd import lesk
from nltk.corpus import stopwords

STOP = set(stopwords.words('english'))
def tokens(text):
    return [w for w in re.findall(r'[a-z]+',text.lower()) if len(w)>2 and w not in STOP]

def signature(s):
    return set(tokens(' '.join([s.definition(), *s.examples(), *(l.name().replace('_',' ') for l in s.lemmas())])))

def disambiguate(word, sentence, method='Context + prior', alpha=4.0, pos=None):
    senses=wn.synsets(word,pos=pos)
    if not senses:
        return {'error':f'No WordNet meanings found for {word!r}.'}
    ctx=Counter(tokens(sentence)); ctx.pop(word.lower(),None)
    total=sum(sum(l.count() for l in s.lemmas())+1 for s in senses)
    scored=[]
    for i,s in enumerate(senses):
        ev=sorted(set(ctx).intersection(signature(s)))
        raw=sum(ctx[w] for w in ev)
        weighted=sum(ctx[w]*(1+math.log(1+len(w))) for w in ev)
        prior=(sum(l.count() for l in s.lemmas())+1)/total
        scored.append(dict(synset=s.name(), definition=s.definition(), evidence=ev,
                           overlap=raw, weighted=weighted, prior=prior, rank=i))
    if method=='NLTK Lesk':
        selected=lesk(sentence.split(),word,pos=pos)
        if selected is not None:
            scored.sort(key=lambda x:(x['synset']!=selected.name(),x['rank']))
    else:
        for item in scored:
            item['score']=({'First sense':lambda x: -x['rank'],
                            'Simplified Lesk':lambda x:x['overlap'],
                            'Weighted Lesk':lambda x:x['weighted'],
                            'Context + prior':lambda x:x['weighted']+alpha*x['prior']}.get(method,lambda x:x['weighted']))(item)
        scored.sort(key=lambda x:(-x['score'],x['rank']))
    best=scored[0]
    return {'word':word,'sentence':sentence,'method':method,'chosen':best,'candidates':scored[:6],
            'warning': 'No matching context evidence; this selection may be unreliable.' if not best['evidence'] else ''}
