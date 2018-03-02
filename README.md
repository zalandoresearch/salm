# Syntax aware language models (SALMs)

Installation, using python 2.7 (python3 not supported).
```
pip install -r requirements.txt
```

Data should be prepared with spaces separating tokens

Example data is in ```./data/coco*```

Training is based on the pytorch word-language model example:
```
python main.py --data <data-directory> --save <save-path> --nsentences <no-train-sentences>
```

Applying a character-level pretrained salm to tag a sentence:

```python
import data
import torch
import particle

corpus = data.Corpus('./data/coco_char_tag')
with open('./models/coco_char.pt', 'rb') as f:
    model = torch.load(f, map_location=lambda storage, loc: storage)
    
# define a SynSiR setup with 100 particles
tagger = particle.CharTagger(model, corpus.dictionary, 100)

sentence = "the man throws the ball to the dog"

for word in sentence.split():
    word += "_"

    word = map(corpus.dictionary.word2idx.__getitem__, list(word))

    # updates return log-likelihood (out-of-sample) of word
    ll = tagger.update(word)
    
print tagger
```