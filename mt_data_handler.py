import copy
import random
from tqdm import tqdm
from input_output_handler import InputOutputHandler

class MTDataHandler:
    def __init__(self, cache_parser, input_tokenizer, target_tokenizer, config, train_phase):
        self.cache_parser = cache_parser
        self.input_tokenizer = input_tokenizer
        self.target_tokenizer = target_tokenizer
        self.query_envs = list(self.cache_parser.query_parser_dict.values())[0].parser_compounds + ['full']

        self.input_output_handler = InputOutputHandler(question_tokenizer=self.input_tokenizer,
                                                       target_tokenizer=self.target_tokenizer,
                                                       config=config)
        self.train_phase = train_phase
        self.config = config


    def prepare_pair(self, question, query, kb_id):
        query_compound_dict = self.cache_parser.get_compounds(query, kb_id)
        masked_queries_list = []
        # compound pairs
        for compound_name, compound_list in query_compound_dict.items():
            if len(compound_list) > 0:
                compound_str = random.choice(compound_list)
                mask_name = f"<{compound_name}>"
                corrupted_query = query.replace(compound_str, mask_name)
                model_input = f"[question] {question} [query] {corrupted_query}"
                model_output = f"{mask_name} {compound_str}"
                masked_queries_list.append({"model_input": model_input,
                                            "model_output": model_output,
                                            "env_name": compound_name,
                                            "question": question,
                                            "query": query,
                                            "kb_id": kb_id})
        # full query
        model_input = f"[question] {question} [query] <full>"
        model_output = f"<full> {query}"
        masked_queries_list.append({"model_input": model_input,
                                    "model_output": model_output,
                                    "env_name": "full",
                                    "question": question,
                                    "query": query,
                                    "kb_id": kb_id})
        return masked_queries_list

    def form_env_datasets(self, input_question_list, input_query_list, kb_id_list):
        all_pairs = []
        for question, query, kb_id in tqdm(zip(input_question_list, input_query_list, kb_id_list), total=len(input_question_list)):
            pairs_list = self.prepare_pair(question, query, kb_id)
            if self.train_phase and self.config['hard_mt']:
                pairs_list = [random.choice(pairs_list)]
            all_pairs += pairs_list

        #collect per env data
        per_env_data_dict = {env: {"input": [], "target": [], "question": [], "query": [], "kb_id": []} for env in self.query_envs}
        for pair in all_pairs:
            env_name = pair['env_name']
            per_env_data_dict[env_name]['input'].append(pair["model_input"])
            per_env_data_dict[env_name]['target'].append(pair["model_output"])
            per_env_data_dict[env_name]['question'].append(pair['question'])
            per_env_data_dict[env_name]['query'].append(pair['query'])
            per_env_data_dict[env_name]['kb_id'].append(pair['kb_id'])

        env_data_dict = dict()
        for env_name in per_env_data_dict:
            if len(per_env_data_dict[env_name]['input']) > 0:
                env_data_dict[env_name] = copy.deepcopy(per_env_data_dict[env_name])
                env_data_dict[env_name]['input'] = self.input_output_handler.prepare_input(input_str_list=per_env_data_dict[env_name]['input'],
                                                                                        input_kb_list=env_data_dict[env_name]['kb_id'],
                                                                                        max_length=256)
                env_data_dict[env_name]['target'] = self.input_output_handler.prepare_output(input_str_list=per_env_data_dict[env_name]['target'],
                                                                                             max_length=128)

        return env_data_dict



