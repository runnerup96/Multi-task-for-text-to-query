from transformers import AutoTokenizer
from collections import OrderedDict

class T5Tokenizer:
    def __init__(self, model_config):
        self.model_config = model_config
        self.hf_tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])
        self.pad_token_id = self.hf_tokenizer.pad_token_id

        self.word2index = OrderedDict()
        self.index2word = OrderedDict()
        self.built_token_mapping()

        self.t5_underscore_token = '▁'
        self.special_tokens_list = ["SOS", "EOS", "UNK", "PAD", " ", self.t5_underscore_token,
                                   '<full>', '[question]', '[query]', '[schema]']
        self.special_tokens_list += self.hf_tokenizer.all_special_tokens
        self.hf_tokenizer.add_tokens(self.special_tokens_list)

    def built_token_mapping(self):
        self.word2index = self.hf_tokenizer.get_vocab()
        self.index2word = OrderedDict({v: k for k, v in self.word2index.items()})

    def __call__(self, text_list, max_length):
        return self.hf_tokenizer(text_list, padding="max_length", max_length=max_length, truncation=True, return_token_type_ids=True)

    def __len__(self):
        return len(self.hf_tokenizer)

    def add_tokens(self, tokens_list):
        self.hf_tokenizer.add_tokens(tokens_list)
        self.built_token_mapping()

    def decode(self, token_list):
        predicted_tokens = []
        for token_id in token_list:
            predicted_token = self.index2word[token_id]
            predicted_token_clean = predicted_token.replace(self.t5_underscore_token, '')
            if len(predicted_token_clean) > 0:
                predicted_tokens.append(predicted_token_clean)
        filtered_tokens = list(filter(lambda x: x not in self.special_tokens_list, predicted_tokens))
        return filtered_tokens