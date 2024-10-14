import torch
import argparse
import yaml
import os
import json
import tqdm

from eval_metrics.eval_sparql import EvalSPARQL
from eval_metrics.eval_sql import EvalSQL
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from mt_data_handler import MTDataHandler
from split_logic.grammar import sparql_parser, atom_and_compound_cache, sql_parser
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from trainer import Trainer
from models.mt_t5_model import MTT5Model
from torch.utils.data import DataLoader
from utils import seed_everything


def run_mt(args):
    seed_everything(42)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
    print('Device: ', DEVICE)

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    print('Run config: ', config)  # log to cluster logs

    model_config = config['model']
    model_name = model_config["used_model"]
    assert model_name in ['t5']
    model_config = model_config[model_name]
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    dev_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['dev']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_query_list = [sample['masked_query'] for sample in train_data]
    dev_query_list = [sample['masked_query'] for sample in dev_data]

    train_kb_ids_list = [sample['kb_id'] for sample in train_data]
    dev_kb_ids_list = [sample['kb_id'] for sample in dev_data]

    QUERY_SPACE_TOKENIZER = QuerySpaceTokenizer(train_query_list, vocab=dataset_vocab_path, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    dev_questions_list = [sample['question'] for sample in dev_data]

    assert config['data']['target_language'] in ['sparql', 'sql']
    target_language = config['data']['target_language']
    evaluator = None
    parser_dict = dict()
    if target_language == 'sparql':
        print('Preparing SPARQL dict!')
        parser_instance = sparql_parser.SPARQLParser(train_query_list)
        parser_dict = {'wikidata': parser_instance}
        print('Prepared SPARQL dict!')
        evaluator = EvalSPARQL(expected_metrics_list=['exact_match', 'graph_match'])
    elif target_language == 'sql':
        db_name2db_path = json.load(open(os.path.join(os.environ['PROJECT_PATH'],
                                                      config['data']['schema'][target_language]['db2db_path']), 'r',
                                         encoding="utf-8"))
        evaluator = EvalSQL(table_name2db_path_dict=db_name2db_path, expected_metrics_list=['exact_match'])
        db2attr_dict = json.load(open(os.path.join(os.environ['PROJECT_PATH'],
                                                   config['data']['schema'][target_language]['db_attributes_dict']), 'r', encoding="utf-8"))
        all_working_samples = train_data + dev_data
        print('Preparing SQL dict!')
        for sample in tqdm.tqdm(all_working_samples, total=len(all_working_samples)):
            db_id = sample['kb_id']
            db_attributes = db2attr_dict[db_id]
            if db_id not in parser_dict:
                parser_instance = sql_parser.SQLParser(db_attributes)
                parser_dict[db_id] = parser_instance

    parser_with_cache = atom_and_compound_cache.AtomAndCompoundCache(parser_dict,
                                                                     query_key_name=None, kb_id_key_name=None,
                                                                     return_compound_list_flag=False,
                                                                     compound_cache_path=config['data']['compound_cache_path'])

    t5_tokenizer = T5Tokenizer(model_config=model_config)
    print('Before tokenizer vocab size: ', len(t5_tokenizer))
    env_list = parser_with_cache.parsers_env_list
    prefix_tokens = [f"<{env_mask}>" for env_mask in env_list]
    dataset_tokens = list(QUERY_SPACE_TOKENIZER.word2index.keys())
    # add dataset tokens + prefix tokens
    t5_tokenizer.special_tokens_list += prefix_tokens
    t5_tokenizer.add_tokens(prefix_tokens + dataset_tokens)
    print('After tokenizer vocab size: ', len(t5_tokenizer))

    mt_data_handler = MTDataHandler(cache_parser=parser_with_cache, input_tokenizer=t5_tokenizer,
                                      target_tokenizer=t5_tokenizer, config=config, train_phase=True)

    print('Preparation train envs data')
    train_env_data = mt_data_handler.form_env_datasets(train_questions_list, train_query_list, train_kb_ids_list)
    print('Preparation val envs data')
    val_env_data = mt_data_handler.form_env_datasets(dev_questions_list, dev_query_list, dev_kb_ids_list)

    train_env_dataloader_dict = {}
    for env in train_env_data:
        train_env_dataset = Text2QueryDataset(tokenized_input_list=train_env_data[env]['input'],
                                              tokenized_target_list=train_env_data[env]['target'],
                                              question_list=train_env_data[env]['question'],
                                              query_list=train_env_data[env]['query'],
                                              tokenizer=t5_tokenizer,
                                              model_type='t5',
                                              dev=DEVICE)
        train_env_dataloader = DataLoader(train_env_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        if config['test_one_batch']:
            train_env_dataloader = [list(train_env_dataloader)[0]]
        train_env_dataloader_dict[env] = train_env_dataloader

    val_env_dataloader_dict = {}
    for env in val_env_data:
        val_env_dataset = Text2QueryDataset(tokenized_input_list=val_env_data[env]['input'],
                                            tokenized_target_list=val_env_data[env]['target'],
                                            question_list=val_env_data[env]['question'],
                                            query_list=val_env_data[env]['query'],
                                            tokenizer=t5_tokenizer,
                                            model_type='t5',
                                            dev=DEVICE)
        val_env_dataloader = DataLoader(val_env_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
        val_env_dataloader_dict[env] = val_env_dataloader

    total_train_steps = len(train_data) // batch_size * model_config['epochs_num']
    t5_model = MTT5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer,
                          compound_types_list=mt_data_handler.query_envs, total_train_steps=total_train_steps)
    trainer = Trainer(model=t5_model, config=config, evaluator=evaluator)

    if config['test_one_batch']:
        trainer.train_with_enviroments(train_env_dataloader_dict, train_env_dataloader_dict)
    else:
        trainer.train_with_enviroments(train_env_dataloader_dict, val_env_dataloader_dict)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    run_mt(args)

