import argparse
from src.wordnet_tool import get_senses
from src.wsd import disambiguate
p=argparse.ArgumentParser(description='WordNet dictionary and WSD')
p.add_argument('word');p.add_argument('-s','--sentence',action='append',default=[])
p.add_argument('--method',default='Context + prior',choices=['Context + prior','Weighted Lesk','Simplified Lesk','NLTK Lesk','First sense'])
p.add_argument('--alpha',type=float,default=4.0);p.add_argument('--pos',default=None)
a=p.parse_args()
for s in get_senses(a.word,a.pos): print(s['synset'],':',s['definition'],'| synonyms:',', '.join(s['synonyms']))
for sentence in a.sentence:
    r=disambiguate(a.word,sentence,a.method,a.alpha,a.pos)
    print('\n',sentence,'\n ->',r.get('chosen',{}).get('synset',r.get('error')),'\n',r.get('chosen',{}).get('definition',''))
    if r.get('warning'): print('Warning:',r['warning'])
