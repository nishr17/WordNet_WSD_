"""Demonstration evaluation; not the 9,570-item SemCor benchmark in the report."""
import csv
from pathlib import Path
from src.wsd import disambiguate
root=Path(__file__).resolve().parents[1]
rows=list(csv.DictReader(open(root/'data'/'demo_contexts.csv',encoding='utf8')))
methods=['First sense','NLTK Lesk','Simplified Lesk','Weighted Lesk','Context + prior']
output=[]
for row in rows:
    for method in methods:
        r=disambiguate(row['word'],row['sentence'],method)
        output.append(dict(**row,method=method,predicted=r['chosen']['synset'],correct=r['chosen']['synset']==row['expected_synset']))
with open(root/'eval'/'results.csv','w',newline='',encoding='utf8') as f:
    w=csv.DictWriter(f,fieldnames=output[0]);w.writeheader();w.writerows(output)
for method in methods:
    part=[x for x in output if x['method']==method]
    print(method,round(100*sum(x['correct'] for x in part)/len(part),1),'% on',len(part),'demo sentences')
