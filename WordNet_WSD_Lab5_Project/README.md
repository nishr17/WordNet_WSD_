# WordNet Word Sense Exploration & Disambiguation — Lab Exercise 5

A runnable **independent reconstruction** based on the supplied lab report, not the original repository. It looks up WordNet definitions, synonyms, examples, hypernyms, hyponyms, antonyms and lemma frequencies, and compares five sense-selection methods.

## Installation

```bash
pip install -r requirements.txt
python setup_nltk.py
streamlit run app.py
```

NLTK data must be downloaded once with internet access. Afterwards, lookup and disambiguation can run offline.

## Command-line examples (different outputs)

```bash
python main.py bank -s "She deposited money at the bank." -s "The fisherman sat on the river bank."
python main.py bat --alpha 0 -s "A bat flew out of the cave." -s "He swung the bat and hit the ball."
python main.py mouse -s "The mouse ran across the floor." -s "I clicked the computer mouse."
```

## Streamlit

**Meanings of a word**: enter any supported English word, filter part of speech, explore definitions and related words, inspect Wu–Palmer similarity, export CSV.

**Meaning in context**: supply multiple sentences for the same ambiguous word. Switch among First sense, NLTK Lesk, Simplified Lesk, Weighted Lesk and Context + prior; change prior weight. Compare results and export CSV. Different words, sentences and methods can produce different outputs.

## Evaluation

```bash
python -m eval.run_eval
```

This evaluates a **small illustrative set**, not the SemCor benchmark described in the PDF. The PDF's reported benchmark accuracies cannot be reproduced without its exact test split and implementation. WSD predictions are heuristic and may be incorrect; the interface warns when it finds no context evidence.
