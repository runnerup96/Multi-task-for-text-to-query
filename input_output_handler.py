import os
import json


class InputOutputHandler:

    def __init__(self, question_tokenizer, target_tokenizer, config):
        self.input_tokenizer = question_tokenizer
        self.target_tokenizer = target_tokenizer
        self.target_language = config['data']['target_language']

        self.schema_config = config['data']['schema'][self.target_language]
        self.add_schema_flag = self.schema_config['add_schema_to_input_flag']
        if self.add_schema_flag:
            if self.target_language == 'sql':
                self.db2attr_dict = json.load(open(os.path.join(os.environ['PROJECT_PATH'],
                                                                self.schema_config['db_attributes_dict']), 'r'))
            elif self.target_language == 'sparql':
                pass

        self.model_name = config['model']['used_model']


    def prepare_output(self, input_str_list, max_length):
        tokenized_output = None
        if self.model_name == 't5':
            tokenized_output = self.input_tokenizer(input_str_list, max_length)
        elif self.model_name == 'vanilla':
            tokenized_output = self.target_tokenizer(input_str_list, max_length)
        return tokenized_output

    def prepare_input(self, input_str_list, input_kb_list, max_length):
        tokenized_input = None
        if self.target_language == 'sparql':
            tokenized_input = self.prepare_sparql_input(input_str_list, input_kb_list, max_length)
        elif self.target_language == 'sql':
            tokenized_input = self.prepare_sql_input(input_str_list, input_kb_list, max_length)
        return tokenized_input


    def prepare_sql_input(self, input_str_list, kb_id_list, max_length):

        input_list = []
        for input_str, kb_id in zip(input_str_list, kb_id_list):
            final_input_str = input_str
            if self.add_schema_flag:
                final_input_tokens = input_str.split() + ['[schema]']
                question_relevant_db_attributes = self.db2attr_dict[kb_id]
                final_input_tokens += question_relevant_db_attributes
                final_input_str = " ".join(final_input_tokens)
            input_list.append(final_input_str)
        tokenized_input_list = self.input_tokenizer(input_list, max_length)
        return tokenized_input_list

    def prepare_sparql_input(self, input_str_list, kb_id_list, max_length):
        input_list = []
        for input_str, kb_id in zip(input_str_list, kb_id_list):
            input_list.append(input_str)
        tokenized_input_list = self.input_tokenizer(input_list, max_length)
        return tokenized_input_list
