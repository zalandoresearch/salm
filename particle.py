import argparse
import torch
from torch.autograd import Variable
import data
from tqdm import tqdm
import math


parser = argparse.ArgumentParser(description='PyTorch PTB Language Model')


parser.add_argument('--data', type=str,
                    help='location of the data corpus')
parser.add_argument('--baseline_data', type=str,
                    help='location of the data corpus')
parser.add_argument('--checkpoint', type=str,
                    help='model checkpoint to use')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--salm', action='store_true',
                    help='use SynSiR')
parser.add_argument('--character', action='store_true',
                    help='character level model/ tagging')
parser.add_argument('--generate', action='store_true',
                    help='generate data instead of computing log-likelihoods')
parser.add_argument('--tag', action='store_true',
                    help='tag a benchmark sentence to check')
parser.add_argument('--complete', action='store_true',
                    help='complete selected sentences')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')


args = parser.parse_args()


def prepare_data():
    print "preparing data..."

    corpus = data.Corpus(args.data)

    plain_corpus = data.Corpus(args.baseline_data)

    test_data = [plain_corpus.dictionary.idx2word[i]
                 for i in plain_corpus.test]

    test_data = [corpus.dictionary.word2idx[x]
                 if x in corpus.dictionary.word2idx.keys()
                 else corpus.dictionary.word2idx["<unk>"]
                 for x in test_data]

    print "data prepared..."
    return corpus, test_data


def prepare_sentences():

    sentences = [[]]
    for word in test_data:
        sentences[-1].append(word)

        if word == corpus.dictionary.word2idx["<eos>"]:
            sentences.append([])

    return sentences


class Tagger:
    def __init__(self, model, dictionary, n_particles=100):
        self.model = model
        self.n_particles = n_particles
        self.hidden = model.init_hidden(n_particles)

        try:
            end_tag = dictionary.word2idx["<eos>"]
        except AttributeError:
            end_tag = dictionary.symbol2idx["<eos>"]

        self.last = torch.LongTensor([end_tag])
        self.last = self.last.repeat(n_particles).unsqueeze(0)
        self.last = Variable(self.last)

        if args.cuda:
            self.last = self.last.cuda()

        self.particles = self.last
        self.dictionary = dictionary
        self.sm = torch.nn.Softmax()

    def update(self, word):

        self.generate()

        output, self.hidden = self.model(self.last, self.hidden)
        self.add_word(word)
        pred = self.sm(output.squeeze())
        weights = pred[:, word]

        self.resample(weights)

        return torch.mean(weights).data[0]

    def add_word(self, word):

        self.last = torch.LongTensor([word])
        self.last = self.last.repeat(self.n_particles).unsqueeze(0)
        self.last = Variable(self.last)
        if args.cuda:
            self.last = self.last.cuda()

        self.particles = torch.cat((self.particles, self.last))

    def resample(self, weights):
        choices = torch.multinomial(weights.unsqueeze(0),
                                    self.n_particles,
                                    replacement=True)

        choices = choices.data.squeeze()

        self.particles = self.particles[:, choices]

        self.hidden = (
            self.hidden[0][:, choices, :],
            self.hidden[1][:, choices, :]
        )

    def generate(self):
        output, self.hidden = self.model(self.last, self.hidden)
        self.last = torch.multinomial(torch.exp(output[0, :, :]), 1)
        self.last = self.last.transpose(1, 0)
        self.particles = torch.cat((self.particles, self.last))

    def refresh(self):
        self.particles = Variable(self.particles[-1:].data)
        self.last = Variable(self.last.data)
        self.hidden = (
            Variable(self.hidden[0].data),
            Variable(self.hidden[1].data)
        )

    def _reprlines(self):
        lines = []
        for i in range(self.n_particles):
            temp = " ".join([self.dictionary.idx2word[i]
                             for i in self.particles[:, i].data.tolist()])
            lines.append(temp)
        return lines

    def __repr__(self):
        return "\n".join(self._reprlines())

    def complete(self):
        output, self.hidden = model(self.last, self.hidden)
        like, argmax = torch.max(output.squeeze())
        self.last = argmax
        self.particles = torch.cat((self.particles, self.last))
        return like


class CharTagger(Tagger):
    def __init__(self, model, dictionary, n_particles):
        Tagger.__init__(self, model, dictionary, n_particles)

    def update(self, word):

        self.generate()
        word_tensor = self.get_word(word)

        output, self.hidden = self.model(torch.cat((self.last,
                                                    word_tensor[:-1, :])),
                                         self.hidden)

        ll = [0 for _ in range(self.n_particles)]
        for i, char in enumerate(word):
            temp = torch.log(self.sm(output[i, :, :]))[:, char]
            for j in range(len(temp)):
                ll[j] += temp[j].data[0]

        weights = [math.exp(x) for x in ll]

        like = sum(weights) / len(weights)

        self.particles = torch.cat((self.particles, word_tensor))
        self.last = word_tensor[-1, :].unsqueeze(0)

        weights = Variable(torch.FloatTensor(weights))
        if args.cuda:
            weights = weights.cuda()

        self.resample(weights)

        return like

    def get_word(self, word):
        word_tensor = torch.LongTensor(word)
        word_tensor = word_tensor.unsqueeze(1).repeat(1, self.n_particles)
        word_tensor = Variable(word_tensor)

        if args.cuda:
            word_tensor = word_tensor.cuda()

        return word_tensor

    def _reprlines(self):
        lines = []
        for i in range(self.n_particles):
            temp = "".join([self.dictionary.idx2word[i]
                             for i in self.particles[:, i].data.tolist()])
            lines.append(temp)
        return lines

    def __repr__(self):
        return "\n".join(self._reprlines())


class Baseline:
    def __init__(self, model, dictionary):
        self.model = model
        self.hidden = model.init_hidden(1)
        self.dictionary = dictionary

        eos = dictionary.word2idx["<eos>"]
        self.last = torch.LongTensor([eos]).unsqueeze(0)
        self.last = Variable(self.last)
        if args.cuda:
            self.last = self.last.cuda()

        self.sm = torch.nn.Softmax()

        self.text = [self.dictionary.word2idx['<eos>']]

    def update(self, word):
        output, self.hidden = self.model(self.last, self.hidden)
        self.add_word(word)
        pred = self.sm(output[0, :, :])
        return pred[0, word].data[0]

    def add_word(self, word):
        self.last = Variable(torch.LongTensor([word]).unsqueeze(0))
        self.text.append(word)

        if args.cuda:
            self.last = self.last.cuda()

    def refresh(self):
        self.last = Variable(self.last.data)
        self.hidden = (
            Variable(self.hidden[0].data),
            Variable(self.hidden[1].data)
        )

    def complete(self):

        output, self.hidden = self.model(self.last, self.hidden)
        like, argmax = output.squeeze().topk(1)

        self.add_word(argmax.data[0])

        return like

    def __repr__(self):
        return ' '.join(map(self.dictionary.idx2word.__getitem__, self.text))

    def generate(self):
        output, self.hidden = self.model(self.last, self.hidden)
        word = torch.multinomial(torch.exp(output[0, :, :]), 1)
        word = word.transpose(1, 0)
        self.add_word(word.squeeze().data[0])


class CharBaseline(Baseline):
    def __init__(self, model, dictionary):
        Baseline.__init__(self, model, dictionary)

    def update(self, word):
        word_tensor = self.get_word(word)
        output, self.hidden = \
            self.model(torch.cat((self.last, word_tensor[:-1, :])), self.hidden)

        ll = 0
        for i, char in enumerate(word):
            temp = torch.log(self.sm(output[i, :, :]))
            ll += temp.data[0, char]
        like = math.exp(ll)

        self.add_word(word[-1])
        return like

    def get_word(self, word):
        word_tensor = torch.LongTensor(word).unsqueeze(1)
        word_tensor = Variable(word_tensor)
        if args.cuda:
            word_tensor = word_tensor.cuda()
        return word_tensor


if __name__ == '__main__':

    corpus, test_data = prepare_data()

    torch.manual_seed(args.seed)
    if torch.cuda.is_available():
        if not args.cuda:
            print("WARNING: You have a CUDA device, "
                  "so you should probably run with --cuda")
        else:
            torch.cuda.manual_seed(args.seed)

    with open(args.checkpoint, 'rb') as f:
        model = torch.load(f, map_location=lambda storage, loc: storage)

    if args.cuda:
        model.cuda()
    else:
        model.cpu()

    model.eval()

    sentences = prepare_sentences()

    if args.generate:
        N = 2
    elif args.tag:
        N = 1000
    else:
        N = 1000

    if args.salm:

        if args.character:
            tagger = CharTagger(model, corpus.dictionary, N)
        else:
            tagger = Tagger(model, corpus.dictionary, N)
    else:

        if args.character:
            tagger = CharBaseline(model, corpus.dictionary)
        else:
            tagger = Baseline(model, corpus.dictionary)

    if args.generate:

        print "generating..."

        for i in range(100):
            tagger.generate()

        print tagger

    elif args.tag:

        sentence = "the man throws the ball to the dog"
        words = sentence.split()

        if args.character:
            for word in words:
                word += "_"

                word = map(corpus.dictionary.word2idx.__getitem__, list(word))

                tagger.update(word)

        else:
            for word in words:
                word = corpus.dictionary.word2idx[word]

                tagger.update(word)

        print tagger

    else:

        if args.character:
            space_id = corpus.dictionary.word2idx["_"]
            words = [[corpus.dictionary.word2idx["<eos>"]]]
            for i, char in enumerate(test_data):
                words[-1].append(char)
                if char == space_id:
                    words.append([])
            words[-1].append(space_id)
        else:
            words = test_data

        ll = []

        for word in tqdm(words):
            ll.append(math.log(tagger.update(word)))
            tagger.refresh()

        print math.exp(-sum(ll) / len(ll))

        with open(str(args.checkpoint).split(".pt")[0] + ".ll.txt", "w") as f:
            f.write("\n".join([str(x) for x in ll]))
