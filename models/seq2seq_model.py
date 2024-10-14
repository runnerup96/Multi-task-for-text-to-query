import numpy as np
import os
import torch
import torch.optim as optim
from torch import nn
from transformers import AutoModel

import lr_scheduler
from models.recurrent_decoder import RecurrentDecoder
from models.seq2seq_attention import Seq2seqAttention
from models.transformer_based_encoder import TransformerBasedEncoder
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer


class Seq2seqModel(nn.Module):
    def __init__(self, model_config: dict, device: str, target_tokenizer: QuerySpaceTokenizer, train_dataset_size):
        super(Seq2seqModel, self).__init__()
        self.model_name = "vanilla"
        self.model_config = model_config
        self.device = device
        self.target_tokenizer = target_tokenizer
        self.batch_size = self.model_config['batch_size']

        hugginface_pretrained_model = self.model_config['model']
        transformer_based_model = AutoModel.from_pretrained(hugginface_pretrained_model)
        trainable_layers_num = self.model_config['n_last_layers2train']
        self.encoder: TransformerBasedEncoder = TransformerBasedEncoder(transformer_based_model,
                                                                        trainable_layers_num).to(self.device)

        decoder_hidden_state_size = self.encoder.bert_module.pooler.dense.weight.shape[0]
        if self.model_config.get('use_pretrained_embeddings', False):
            trained_embeddings = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                                         self.model_config['pretrained_embeddings_path']))
            decoder_input_size = trained_embeddings.shape[-1]
            self.decoder: RecurrentDecoder = RecurrentDecoder(input_size=decoder_input_size,
                                                              hidden_size=decoder_hidden_state_size,
                                                              vocab_size=len(target_tokenizer.word2index),
                                                              trained_embeddings=trained_embeddings).to(self.device)
        else:
            self.decoder: RecurrentDecoder = RecurrentDecoder(input_size=self.model_config['embeddings_size'],
                                                              hidden_size=decoder_hidden_state_size,
                                                              vocab_size=len(target_tokenizer.word2index),
                                                              trained_embeddings=None).to(self.device)

        self.softmax = nn.LogSoftmax(dim=1).to(self.device)

        self.optimizer = optim.Adam(
            [
                {'params': self.encoder.parameters(), 'lr': self.model_config['bert_finetune_rate']},
                {'params': self.decoder.parameters()},
            ], lr=self.model_config['learning_rate'])

        if self.model_config['enable_attention']:
            self.attention_module = Seq2seqAttention().to(self.device)
            self.vocab_projection_layer = nn.Linear(2 * decoder_hidden_state_size, len(target_tokenizer.word2index)).to(
                self.device)
        else:
            self.vocab_projection_layer = nn.Linear(decoder_hidden_state_size, len(target_tokenizer.word2index)).to(
                self.device)

        self.optimizer.add_param_group({"params": self.vocab_projection_layer.parameters()})
        self.optimizer_scheduler = lr_scheduler.InverseSquareRootScheduler(optimizer=self.optimizer,
                                                                           warmup_init_lrs=[
                                                                               self.model_config['bert_warmup_init_finetuning_learning_rate'],
                                                                               self.model_config['warm_up_init_learning_rate'],
                                                                               self.model_config['warm_up_init_learning_rate']],
                                                                           num_warmup_steps=[self.model_config['warmup_steps'], self.model_config['warmup_steps'], self.model_config['warmup_steps']],
                                                                           num_steps=int(train_dataset_size // self.model_config['batch_size'] * self.model_config['epochs_num']))
        self.criterion = nn.NLLLoss()
        self.teacher_forcing_ratio = 0.5

    def train_on_batch(self, input_data, target_data=None):
        self.encoder.enable_bert_layers_training()
        self.optimizer.zero_grad()

        encoder_output = self.encoder(input_data)

        encoder_states = encoder_output['last_hiddens']
        pooler = encoder_output['pooler']

        decoder_input = torch.tensor([[0] * self.batch_size],
                                     dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
        decoder_hidden = pooler.view(1, self.batch_size, -1)
        decoder_cell_state = torch.zeros(1, self.batch_size, decoder_hidden.shape[-1], device=self.device)

        target_tensor = target_data['input_ids'].view(self.batch_size, self.target_tokenizer.max_sent_len, 1)

        target_length = target_tensor.shape[1]
        loss = 0.0
        for idx in range(target_length):
            decoder_output, decoder_hidden, decoder_cell_state = self.decoder(input_data=decoder_input,
                                                                              hidden_state=decoder_hidden,
                                                                              cell_state=decoder_cell_state,
                                                                              batch_size=self.batch_size)
            # Добавляем взвешивание механизмом внимания
            if self.model_config['enable_attention']:
                # decoder_output - ([1, batch_size, dim])
                weighted_decoder_output = self.attention_module(decoder_output.squeeze(dim=0), encoder_states)
                # weighted_decoder_output - ([batch_size, dim])
                concated_attn_decoder = torch.cat([decoder_output.squeeze(dim=0), weighted_decoder_output], dim=1)
                # concated_attn_decoder - ([batch_size, 2 * dim])
                linear_vocab_proj = self.vocab_projection_layer(concated_attn_decoder)
                # concated_attn_decoder - ([batch_size, vocab_size])
            else:
                linear_vocab_proj = self.vocab_projection_layer(decoder_output)


            target_vocab_distribution = self.softmax(linear_vocab_proj)

            use_teacher_forcing = True if np.random.random() < self.teacher_forcing_ratio else False
            if use_teacher_forcing:
                decoder_input = target_tensor[:, idx, :].reshape(1, self.batch_size, 1)
            else:
                _, top_index = target_vocab_distribution.topk(1)
                decoder_input = top_index.reshape(1, self.batch_size, 1)

            loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())

        loss = loss / target_length
        loss.backward()
        self.optimizer.step()
        self.optimizer_scheduler.step()

        return loss.item()

    def evaluate_batch(self, input_data, target_data=None):
        self.encoder.disable_bert_training()
        result_dict = dict()

        with torch.no_grad():
            encoder_output = self.encoder(input_data)

            encoder_states = encoder_output['last_hiddens']
            pooler = encoder_output['pooler']

            decoder_input = torch.tensor([[0] * self.batch_size],
                                         dtype=torch.long, device=self.device).view(1, self.batch_size, 1)
            decoder_hidden = pooler.view(1, self.batch_size, -1)
            decoder_cell_state = torch.zeros(1, self.batch_size, decoder_hidden.shape[-1], device=self.device)

            decoder_result_list = []
            loss = 0.0
            hidden_states_list = []
            for idx in range(self.target_tokenizer.max_sent_len):
                decoder_output, decoder_hidden, decoder_cell_state = self.decoder(input_data=decoder_input,
                                                                                  hidden_state=decoder_hidden,
                                                                                  cell_state=decoder_cell_state,
                                                                                  batch_size=self.batch_size)

                if self.model_config['enable_attention']:
                    # decoder_output - ([1, batch_size, dim])
                    weighted_decoder_output = self.attention_module(decoder_output.squeeze(dim=0), encoder_states)
                    # weighted_decoder_output - ([batch_size, dim])
                    concated_attn_decoder = torch.cat([decoder_output.squeeze(dim=0), weighted_decoder_output], dim=1)
                    # concated_attn_decoder - ([batch_size, 2 * dim])
                    linear_vocab_proj = self.vocab_projection_layer(concated_attn_decoder)
                    # concated_attn_decoder - ([batch_size, vocab_size])
                else:
                    linear_vocab_proj = self.vocab_projection_layer(decoder_output)

                target_vocab_distribution = self.softmax(linear_vocab_proj)
                _, top_index = target_vocab_distribution.topk(1)
                decoder_input = top_index.reshape(1, self.batch_size, 1)

                if target_data:
                    target_tensor = target_data['input_ids'].view(self.batch_size,
                                                                  self.target_tokenizer.max_sent_len, 1)
                    loss += self.criterion(target_vocab_distribution.squeeze(), target_tensor[:, idx, :].squeeze())
                decoder_result_list.append(list(decoder_input.flatten().cpu().numpy()))

                #decoder_hidden - (batch_size, hidden_dim) * max_sent_len -> [max_sent_len, batch_size, hidden_dim]
                hidden_states_list.append(decoder_hidden)

            if loss != 0:
                loss = loss / self.target_tokenizer.max_sent_len
                result_dict['loss'] = loss.item()
            decoder_result_transposed = np.array(decoder_result_list).T
            decoder_result_transposed_lists = [list(array) for array in decoder_result_transposed]

            decoded_query_list = []
            for sample in decoder_result_transposed_lists:
                decoded_query_tokens = self.target_tokenizer.decode(sample)
                query = " ".join(decoded_query_tokens)
                decoded_query_list.append(query)

            result_dict['predicted_query'] = decoded_query_list
            result_dict['predicted_hidden_states'] = torch.stack(hidden_states_list).cpu().numpy()


        return result_dict
