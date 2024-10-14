import json
import os
import argparse

import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from models.seq2seq_model import Seq2seqModel
from models.t5_model import T5Model
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from trainer import Trainer
from eval_metrics.eval_sql import EvalSQL
from eval_metrics.eval_sparql import EvalSPARQL
from input_output_handler import InputOutputHandler
from utils import seed_everything


def run_erm(args):
    seed_everything(42)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'
    print('DEVICE: ', DEVICE)

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)

    print('Run config: ', config)  # log to cluster logs

    model_config = config['model']
    model_name = model_config["used_model"]
    assert model_name in ['vanilla', 't5']
    model_config = model_config[model_name]
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    dev_data = json.load(open(os.path.join(os.environ['PROJECT_PATH'], config['data']['dev']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_query_list = [sample['masked_query'] for sample in train_data]
    dev_query_list = [sample['masked_query'] for sample in dev_data]

    QUERY_SPACE_TOKENIZER = QuerySpaceTokenizer(train_query_list, vocab=dataset_vocab_path, pad_flag=True)

    train_questions_list = [sample['question'] for sample in train_data]
    dev_questions_list = [sample['question'] for sample in dev_data]

    prefixed_train_questions_list = [f"[question] {sample['question']} [query] <full>" for sample in train_data]
    prefixed_dev_questions_list = [f"[question] {sample['question']} [query] <full>" for sample in dev_data]
    prefixed_train_query_list = [f"<full> {sample['masked_query']}" for sample in train_data]
    prefixed_dev_query_list = [f"<full> {sample['masked_query']}" for sample in dev_data]

    assert config['data']['target_language'] in ['sparql', 'sql']
    target_language = config['data']['target_language']
    evaluator = None
    if target_language == 'sql':
        db_name2db_path = json.load(open(os.path.join(os.environ['PROJECT_PATH'],
                                                      config['data']['schema'][target_language]['db2db_path']), 'r', encoding="utf-8"))
        evaluator = EvalSQL(table_name2db_path_dict=db_name2db_path, expected_metrics_list=['exact_match'])
    elif target_language == 'sparql':
        evaluator = EvalSPARQL(expected_metrics_list=['exact_match', 'graph_match'])

    trainer = None
    target_tokenizer = None
    input_tokenizer = None
    if model_name == 't5':

        t5_tokenizer = T5Tokenizer(model_config=model_config)
        print('Before tokenizer vocab size: ', len(t5_tokenizer))
        t5_tokenizer.add_tokens(list(QUERY_SPACE_TOKENIZER.word2index.keys()))
        print('After tokenizer vocab size: ', len(t5_tokenizer))

        total_train_steps = len(train_data) // batch_size * model_config['epochs_num']
        t5_model = T5Model(model_config=model_config, device=DEVICE, tokenizer=t5_tokenizer,
                           total_train_steps=total_train_steps)
        trainer = Trainer(model=t5_model, config=config, evaluator=evaluator)
        input_tokenizer = t5_tokenizer
        target_tokenizer = t5_tokenizer

    elif model_name == 'vanilla':
        input_tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])

        seq2seq = Seq2seqModel(model_config=model_config, device=DEVICE, target_tokenizer=QUERY_SPACE_TOKENIZER,
                               train_dataset_size=len(train_questions_list))
        trainer = Trainer(model=seq2seq, config=config, evaluator=evaluator)

        input_tokenizer = input_tokenizer
        target_tokenizer = QUERY_SPACE_TOKENIZER

    input_output_handler = InputOutputHandler(question_tokenizer=input_tokenizer,
                                              target_tokenizer=target_tokenizer, config=config)
    train_kb_id_list = [sample['kb_id'] for sample in train_data]
    dev_kb_id_list = [sample['kb_id'] for sample in dev_data]

    train_tokenized_questions_list = input_output_handler.prepare_input(
        input_str_list=prefixed_train_questions_list,
        input_kb_list=train_kb_id_list, max_length=256)
    dev_tokenized_questions_list = input_output_handler.prepare_input(
            input_str_list=prefixed_dev_questions_list, input_kb_list=dev_kb_id_list, max_length=256)
    train_tokenized_query_list = input_output_handler.prepare_output(prefixed_train_query_list, max_length=128)
    dev_tokenized_query_list = input_output_handler.prepare_output(prefixed_dev_query_list, max_length=128)

    train_dataset = Text2QueryDataset(tokenized_input_list=train_tokenized_questions_list,
                                      tokenized_target_list=train_tokenized_query_list,
                                      question_list=train_questions_list,
                                      query_list=train_query_list,
                                      tokenizer=target_tokenizer,
                                      model_type=model_name,
                                      dev=DEVICE)

    dev_dataset = Text2QueryDataset(tokenized_input_list=dev_tokenized_questions_list,
                                    tokenized_target_list=dev_tokenized_query_list,
                                    question_list=dev_questions_list,
                                    query_list=dev_query_list,
                                    tokenizer=target_tokenizer,
                                    model_type=model_name,
                                    dev=DEVICE)

    train_dataloader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True, drop_last=False)
    dev_dataloader = DataLoader(dev_dataset, batch_size=batch_size, shuffle=True, drop_last=False)

    # если хотим проверить на 1ом батче
    if config['test_one_batch']:
        train_dataloader_sample = [list(train_dataloader)[0]]
        trainer.train(train_dataloader_sample, train_dataloader_sample)
    else:
        trainer.train(train_dataloader, dev_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    run_erm(args)
