import nltk
for resource in ['wordnet','omw-1.4','stopwords']:
    print('Downloading',resource)
    nltk.download(resource)
print('Setup complete.')
