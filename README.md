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

## License

The MIT License (MIT) Copyright (c) 2018 Zalando SE

Permission is hereby granted, free of charge, to any person obtaining a copy of this software and associated documentation files (the "Software"), to deal in the Software without restriction, including without limitation the rights to use, copy, modify, merge, publish, distribute, sublicense, and/or sell copies of the Software, and to permit persons to whom the Software is furnished to do so, subject to the following conditions:

The above copyright notice and this permission notice shall be included in all copies or substantial portions of the Software.

THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY, FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM, OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE SOFTWARE.
