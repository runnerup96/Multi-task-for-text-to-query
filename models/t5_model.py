import torch
from torch import nn
from torch.nn import CrossEntropyLoss
from transformers import T5ForConditionalGeneration
from transformers.optimization import Adafactor, AdamW, get_linear_schedule_with_warmup


class T5Model(nn.Module):
    def __init__(self, model_config, device, tokenizer, total_train_steps=None):
        super(T5Model, self).__init__()
        self.model_name = "t5"
        self.model_config = model_config
        self.device = device
        self.tokenizer = tokenizer
        self.batch_size = self.model_config['batch_size']

        hugginface_pretrained_model = self.model_config['model']
        self.t5_model = T5ForConditionalGeneration.from_pretrained(hugginface_pretrained_model).to(self.device)
        self.t5_model.resize_token_embeddings(len(tokenizer))

        self.lr_scheduler = None
        self.optimizer = None
        assert self.model_config['optimizer'] in ['adafactor', 'adam']
        if self.model_config['optimizer'] == 'adafactor':
            # as in https://arxiv.org/pdf/1910.10683.pdf for fine-tuning
            self.optimizer = Adafactor(self.t5_model.parameters(), lr=self.model_config['learning_rate'], relative_step=False)
        elif self.model_config['optimizer'] == 'adam':
            self.optimizer = AdamW(self.t5_model.parameters(), lr=self.model_config['learning_rate'])
            self.lr_scheduler = get_linear_schedule_with_warmup(self.optimizer, num_warmup_steps=int(total_train_steps*0.01),
                                                                num_training_steps=total_train_steps)
        self.criterion = CrossEntropyLoss(ignore_index=-100)

    def train_on_batch(self, input_data, target_data):
        self.t5_model.train()
        self.optimizer.zero_grad()

        input_ids, attention_mask = input_data['input_ids'], input_data['attention_mask']
        target_ids = target_data['input_ids']

        loss = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids).loss

        loss.backward()
        self.optimizer.step()
        if self.lr_scheduler:
            self.lr_scheduler.step()

        return loss.item()

    def evaluate_batch(self, input_data, target_data):
        self.t5_model.eval()
        result_dict = dict()
        input_ids, attention_mask = input_data['input_ids'], input_data['attention_mask']
        target_ids = target_data['input_ids']
        with torch.no_grad():
            output = self.t5_model(input_ids=input_ids, attention_mask=attention_mask, labels=target_ids)
        loss = output.loss
        decoder_result_list = torch.argmax(output.logits, dim=-1).cpu().numpy()
        result_dict['loss'] = loss.item()

        decoded_query_list = []
        for sample in decoder_result_list:
            decoded_query_tokens = self.tokenizer.decode(sample)
            query = " ".join(decoded_query_tokens)
            decoded_query_list.append(query)

        result_dict['predicted_query'] = decoded_query_list
        return result_dict
