import streamlit as st
import pandas as pd
import plotly.express as px
from src.wordnet_tool import get_senses, similarity_matrix
from src.wsd import disambiguate

st.set_page_config(page_title='WordNet Word Sense Explorer',layout='wide')
st.title('📖 WordNet Word Sense Explorer')
st.caption('Lab 5 · Synonyms, definitions and word-sense disambiguation · NLTK WordNet')
t1,t2=st.tabs(['Meanings of a word','Meaning in context'])
with t1:
    word=st.text_input('Word to explore','bank')
    pos_label=st.selectbox('Part of speech',['All','Noun','Verb','Adjective','Adverb'])
    pos={'All':None,'Noun':'n','Verb':'v','Adjective':'a','Adverb':'r'}[pos_label]
    if st.button('Explore meanings',type='primary'):
        try:
            records=get_senses(word,pos)
            st.metric('Meanings',len(records))
            if not records: st.warning('No meanings found.')
            else:
                for i,r in enumerate(records,1):
                    with st.expander(f"{i}. {r['synset']} — {r['definition']}",expanded=i<=2):
                        st.write('**Synonyms:**',', '.join(r['synonyms']) or 'None')
                        st.write('**Examples:**','; '.join(r['examples']) or 'None')
                        st.write('**Broader:**',', '.join(r['hypernyms']) or 'None')
                        st.write('**Narrower:**',', '.join(r['hyponyms']) or 'None')
                        st.write('**Antonyms:**',', '.join(r['antonyms']) or 'None')
                        st.write('**SemCor lemma frequency:**',r['frequency'])
                st.download_button('Download meanings CSV',pd.DataFrame(records).to_csv(index=False),'wordnet_meanings.csv')
                labels,matrix=similarity_matrix(word,'n' if pos is None else pos)
                if labels:
                    with st.expander('Wu–Palmer sense similarity'):
                        st.plotly_chart(px.imshow(matrix,x=labels,y=labels,color_continuous_scale='Blues',zmin=0,zmax=1),use_container_width=True)
        except LookupError:
            st.error('NLTK data missing. Run: python setup_nltk.py')
with t2:
    word2=st.text_input('Ambiguous word','bank',key='context_word')
    sentences=st.text_area('One sentence per line', 'She deposited money at the bank.\nThe fisherman rested on the river bank.\nThe bank approved the loan.\nThey sat on the bank and watched the water.',height=150)
    method=st.selectbox('Method',['Context + prior','Weighted Lesk','Simplified Lesk','NLTK Lesk','First sense'])
    alpha=st.slider('Sense-frequency prior weight',0.0,8.0,4.0,0.25)
    if st.button('Disambiguate sentences',type='primary'):
        try:
            rows=[]
            for sentence in [x.strip() for x in sentences.splitlines() if x.strip()]:
                r=disambiguate(word2,sentence,method,alpha)
                if 'error' in r: st.error(r['error']); continue
                st.subheader(sentence)
                st.success(f"{r['chosen']['synset']}: {r['chosen']['definition']}")
                st.write('Evidence:',', '.join(r['chosen']['evidence']) or 'No matching words')
                if r['warning']: st.warning(r['warning'])
                with st.expander('Other possible meanings'):
                    st.dataframe(pd.DataFrame(r['candidates']),use_container_width=True)
                rows.append({'sentence':sentence,'method':method,'sense':r['chosen']['synset'],'definition':r['chosen']['definition'],'evidence':', '.join(r['chosen']['evidence'])})
            if rows:
                st.download_button('Download results CSV',pd.DataFrame(rows).to_csv(index=False),'wsd_results.csv')
                if len(rows)>1:
                    st.subheader('Compare all five methods')
                    comparison=[]
                    for sentence in [x['sentence'] for x in rows]:
                        row={'Sentence':sentence}
                        for m in ['First sense','NLTK Lesk','Simplified Lesk','Weighted Lesk','Context + prior']:
                            row[m]=disambiguate(word2,sentence,m,alpha)['chosen']['synset']
                        comparison.append(row)
                    st.dataframe(pd.DataFrame(comparison),use_container_width=True)
        except LookupError:
            st.error('NLTK data missing. Run: python setup_nltk.py')
