import json
from collections import OrderedDict


class QuerySpaceTokenizer:
    def __init__(self, query_list, vocab, pad_flag):
        self.pad_flag = pad_flag
        self.word2count = OrderedDict()
        self.index2word = OrderedDict({0: "SOS", 1: "EOS", 2: "UNK", 3: "PAD", 4: " ",
                           5: "[question]", 6: "[query]", 7: "[schema]",
                           8: "<full>"})
        self.word2index = OrderedDict({v:k for k, v in self.index2word.items()})
        self.n_words = len(self.word2index)
        self.max_sent_len = -1
        self.special_tokens_list = list(self.index2word.values())

        assert type(vocab) in [list, str]
        if type(vocab) is str:
            token_list = json.load(open(vocab, 'r'))
            self.load_vocab(token_list)
        elif type(vocab) is list:
            self.load_vocab(vocab)

        for sent in query_list:
            sent_words_amount = len(sent.split())
            if sent_words_amount > self.max_sent_len:
                self.max_sent_len = sent_words_amount

        self.max_sent_len += 2# always add SOS/EOS

        print(f'Code tokenizer fitted - {len(self.word2index)} tokens')

    def load_vocab(self, token_list):
        for token in token_list:
            if token not in self.word2index:
                self.word2index[token] = self.n_words
                self.word2count[token] = 1
                self.index2word[self.n_words] = token
                self.n_words += 1
            else:
                self.word2count[token] += 1

    def pad_sent(self, token_ids_list):
        if len(token_ids_list) < self.max_sent_len:
            padded_token_ids_list = token_ids_list + [self.word2index['PAD']] * (self.max_sent_len - len(token_ids_list))
        else:
            padded_token_ids_list = token_ids_list[:self.max_sent_len - 1] + [self.word2index['EOS']]
        return padded_token_ids_list

    def __call__(self, sentence):
        tokenized_data = self.tokenize(sentence)
        if self.pad_flag:
            tokenized_data = self.pad_sent(tokenized_data)
        return tokenized_data

    def tokenize(self, sentence):
        tokenized_data = []
        tokenized_data.append(self.word2index['SOS'])
        for word in sentence.split():
            if word in self.word2index:
                tokenized_data.append(self.word2index[word])
            else:
                tokenized_data.append(self.word2index['UNK'])
        tokenized_data.append(self.word2index['EOS'])
        return tokenized_data

    def decode(self, token_list):
        predicted_tokens = []

        for token_id in token_list:
            predicted_token = self.index2word[token_id]
            predicted_tokens.append(predicted_token)
        filtered_tokens = list(filter(lambda x: x not in self.special_tokens_list, predicted_tokens))

        return filtered_tokens