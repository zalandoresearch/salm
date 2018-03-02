import argparse
import time
import math

import torch
import torch.nn as nn
from torch.autograd import Variable

import data
import model

parser = argparse.ArgumentParser(description='PyTorch PennTreeBank RNN/LSTM Language Model')
parser.add_argument('--data', type=str, default='./data/penn',
                    help='location of the data corpus')
parser.add_argument('--model', type=str, default='LSTM',
                    help='type of recurrent net (RNN_TANH, RNN_RELU, LSTM, GRU)')
parser.add_argument('--emsize', type=int, default=200,
                    help='size of word embeddings')
parser.add_argument('--nhid', type=int, default=200,
                    help='number of hidden units per layer')
parser.add_argument('--nlayers', type=int, default=2,
                    help='number of layers')
parser.add_argument('--lr', type=float, default=20,
                    help='initial learning rate')
parser.add_argument('--clip', type=float, default=0.25,
                    help='gradient clipping')
parser.add_argument('--epochs', type=int, default=40,
                    help='upper epoch limit')
parser.add_argument('--batch_size', type=int, default=20, metavar='N',
                    help='batch size')
parser.add_argument('--bptt', type=int, default=35,
                    help='sequence length')
parser.add_argument('--dropout', type=float, default=0.2,
                    help='dropout applied to layers (0 = no dropout)')
parser.add_argument('--tied', action='store_true',
                    help='tie the word embedding and softmax weights')
parser.add_argument('--seed', type=int, default=1111,
                    help='random seed')
parser.add_argument('--cuda', action='store_true',
                    help='use CUDA')
parser.add_argument('--log-interval', type=int, default=200, metavar='N',
                    help='report interval')
parser.add_argument('--save', type=str,  default='./models/penn.pt',
                    help='path to save the final model')
parser.add_argument('--nsentences', type=int, default=42068,
                    help='no. of sentences to train on')

args = parser.parse_args()

torch.manual_seed(args.seed)
if torch.cuda.is_available():
    if not args.cuda:
        print("WARNING: You have a CUDA device, so you should probably run with --cuda")
    else:
        torch.cuda.manual_seed(args.seed)

print "getting data..."
corpus = data.Corpus(args.data)

eval_batch_size = 10

print "batching..."

stops = [i for i in range(len(corpus.train))
         if corpus.train[i] == corpus.dictionary.word2idx["<eos>"]]

last = stops[args.nsentences - 1]
corpus.train = corpus.train[:last]

train_data = data.batchify(corpus.train, args.batch_size, args.cuda)
valid_data = data.batchify(corpus.valid, eval_batch_size, args.cuda)
test_data = data.batchify(corpus.test, eval_batch_size, args.cuda)

print "getting model..."

ntokens = len(corpus.dictionary)
lm = model.RNNModel(args.model, ntokens, args.emsize, args.nhid, args.nlayers, args.dropout, args.tied)


if args.cuda:
    lm.cuda()

criterion = nn.CrossEntropyLoss()


def repackage_hidden(h):
    if type(h) == Variable:
        return Variable(h.data)
    else:
        return tuple(repackage_hidden(v) for v in h)


def evaluate(data_source):
    lm.eval()
    total_loss = 0
    ntokens = len(corpus.dictionary)
    hidden = lm.init_hidden(eval_batch_size)
    for i in range(0, data_source.size(0) - 1, args.bptt):
        dat_, targets = data.get_batch(data_source, i, args.bptt, evaluation=True)
        output, hidden = lm(dat_, hidden)
        output_flat = output.view(-1, ntokens)
        total_loss += len(dat_) * criterion(output_flat, targets).data
        hidden = repackage_hidden(hidden)
    return total_loss[0] / len(data_source)


def train():
    lm.train()

    total_loss = 0
    start_time = time.time()
    hidden = lm.init_hidden(args.batch_size)
    for batch, i in enumerate(range(0, train_data.size(0) - 1, args.bptt)):

        dat_, targets = data.get_batch(train_data, i, args.bptt)

        hidden = repackage_hidden(hidden)
        lm.zero_grad()

        output, hidden = lm(dat_, hidden)
        loss = criterion(output.view(-1, len(corpus.dictionary)), targets)

        loss.backward()

        torch.nn.utils.clip_grad_norm(lm.parameters(), args.clip)
        for p in lm.parameters():
            p.data.add_(-lr, p.grad.data)

        total_loss += loss.data

        if batch % args.log_interval == 0 and batch > 0:
            cur_loss = total_loss[0] / args.log_interval
            elapsed = time.time() - start_time
            print('| epoch {:3d} | {:5d}/{:5d} batches | lr {:02.2f} | ms/batch {:5.2f} | '
                    'loss {:5.2f} | ppl {:8.2f}'.format(
                epoch, batch, len(train_data) // args.bptt, lr,
                elapsed * 1000 / args.log_interval, cur_loss, math.exp(cur_loss)))
            total_loss = 0
            start_time = time.time()

lr = args.lr
best_val_loss = None

print "training..."
try:
    for epoch in range(1, args.epochs+1):
        epoch_start_time = time.time()
        train()

        val_loss = evaluate(valid_data)

        print('-' * 89)
        print('| end of epoch {:3d} | time: {:5.2f}s | valid loss {:5.2f} | '
                'valid ppl {:8.2f}'.format(epoch, (time.time() - epoch_start_time),
                                           val_loss, math.exp(val_loss)))
        print('-' * 89)

        if not best_val_loss or val_loss < best_val_loss:
            with open(args.save, 'wb') as f:
                torch.save(lm, f)
            best_val_loss = val_loss
        else:
            lr /= 4.0

except KeyboardInterrupt:
    print('-' * 89)
    print('Exiting from training early')

with open(args.save, 'rb') as f:
    lm = torch.load(f)


test_loss = evaluate(test_data)


print('=' * 89)
print('| End of training | test loss {:5.2f} | test ppl {:8.2f}'.format(
    test_loss, math.exp(test_loss)))
print('=' * 89)
