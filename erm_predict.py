import json
import os
import argparse
import numpy as np
import torch
import yaml
from torch.utils.data import DataLoader
from transformers import AutoTokenizer

from eval_metrics.eval_sparql import EvalSPARQL
from eval_metrics.eval_sql import EvalSQL
from input_output_handler import InputOutputHandler
from models.seq2seq_model import Seq2seqModel
from models.t5_model import T5Model
from predictor import Predictor
from target_tokenizers.query_space_tokenizer import QuerySpaceTokenizer
from target_tokenizers.t5_tokenizer import T5Tokenizer
from text2query_dataset import Text2QueryDataset
from utils import seed_everything

def erm_predict(args):
    seed_everything(42)

    if torch.cuda.is_available():
        DEVICE = "cuda"
    else:
        DEVICE = 'cpu'

    config_name = args.config_name
    config = yaml.load((open(os.path.join(os.environ['PROJECT_PATH'], "configs", config_name), 'r', encoding="utf-8")),
                       Loader=yaml.Loader)
    trained_model_path = os.path.join(os.environ['PROJECT_PATH'], config['save_model_path'],
                                      config['inference_model_name'].split("/")[0])

    model_config = json.load(open(os.path.join(trained_model_path, 'model_config.json'), 'r'))
    model_name = model_config["model_name"]
    assert model_name in ['vanilla', 't5']
    batch_size = model_config['batch_size']

    train_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['train']), 'r', encoding="utf-8"))
    test_data = json.load(
        open(os.path.join(os.environ['PROJECT_PATH'], config['data']['test']), 'r', encoding="utf-8"))
    dataset_vocab_path = os.path.join(os.environ['PROJECT_PATH'], config['data']['dataset_vocab'])

    train_query_list = [sample['masked_query'] for sample in train_data]
    test_query_list = [sample['masked_query'] for sample in test_data]

    QUERY_SPACE_TOKENIZER = QuerySpaceTokenizer(train_query_list, vocab=dataset_vocab_path, pad_flag=True)

    test_questions_list = [sample['question'] for sample in test_data]

    prefixed_test_questions_list = [f"[question] {sample['question']} [query] <full>" for sample in test_data]
    prefixed_test_query_list = [f"<full> {sample['masked_query']}" for sample in test_data]

    assert config['data']['target_language'] in ['sparql', 'sql']
    target_language = config['data']['target_language']
    evaluator = None
    if target_language == 'sql':
        db_name2db_path = json.load(open(os.path.join(os.environ['PROJECT_PATH'],
                                                      config['data']['schema'][target_language]['db2db_path']), 'r', encoding="utf-8"))
        evaluator = EvalSQL(table_name2db_path_dict=db_name2db_path)
    elif target_language == 'sparql':
        evaluator = EvalSPARQL()

    predictor = None
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
        t5_state = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                           config['save_model_path'], config['inference_model_name']),
                              map_location=DEVICE)['model_state_dict']
        t5_model.load_state_dict(t5_state)
        predictor = Predictor(model=t5_model, config=config, evaluator=evaluator)
        input_tokenizer = t5_tokenizer
        target_tokenizer = t5_tokenizer

    elif model_name == 'vanilla':
        input_tokenizer = AutoTokenizer.from_pretrained(model_config['tokenizer'])

        seq2seq = Seq2seqModel(model_config=model_config, device=DEVICE, target_tokenizer=QUERY_SPACE_TOKENIZER,
                               train_dataset_size=0)
        seq2seq_state = torch.load(os.path.join(os.environ['PROJECT_PATH'],
                                           config['save_model_path'], config['inference_model_name']),
                              map_location=DEVICE)['model_state_dict']
        seq2seq.load_state_dict(seq2seq_state)
        predictor = Predictor(model=seq2seq, config=config, evaluator=evaluator)
        input_tokenizer = input_tokenizer
        target_tokenizer = QUERY_SPACE_TOKENIZER

    input_output_handler = InputOutputHandler(question_tokenizer=input_tokenizer,
                                              target_tokenizer=target_tokenizer, config=config)
    test_kb_id_list = [sample['kb_id'] for sample in test_data]

    test_tokenized_questions_list = input_output_handler.prepare_input(
        input_str_list=prefixed_test_questions_list, input_kb_list=test_kb_id_list, max_length=256)
    test_tokenized_query_list = input_output_handler.prepare_output(prefixed_test_query_list, max_length=128)

    test_dataset = Text2QueryDataset(tokenized_input_list=test_tokenized_questions_list,
                                     tokenized_target_list=test_tokenized_query_list,
                                     question_list=test_questions_list,
                                     query_list=test_query_list,
                                     tokenizer=target_tokenizer,
                                     model_type=model_name,
                                     dev=DEVICE)

    test_dataloader = DataLoader(test_dataset, batch_size=batch_size, shuffle=False, drop_last=False)

    predictor.predict(test_dataloader)


if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument("--config_name", type=str, required=True)
    args = parser.parse_args()
    erm_predict(args)
