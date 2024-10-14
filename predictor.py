import os

from tqdm import tqdm

import utils

class Predictor:
    def __init__(self, model, config, evaluator):
        self.model = model
        self.config = config
        self.model_train_type = self.config['model_type']
        self.evaluator = evaluator
        assert self.model_train_type in ['erm', 'mt']

    def predict(self, dataloader):
        test_metrics_dict = {key: 0 for key in self.evaluator.used_eval_functions}
        input_questions = []
        predicted_queries, true_queries = [], []

        self.model.eval()
        for batch in tqdm(dataloader):
            input_data, target_data = batch['input'], batch['target']

            if self.model_train_type == 'mt':
                eval_result = self.model.evaluate_batch(input_data, target_data, "full")
            else:
                eval_result = self.model.evaluate_batch(input_data, target_data)

            pred_metrics = self.evaluator.calculate_batch_metrics(eval_result['predicted_query'],
                                                                  target_data['original_query'])
            for key in pred_metrics:
                test_metrics_dict[key] += pred_metrics[key]

            input_questions += input_data['original_question']
            predicted_queries += eval_result['predicted_query']
            true_queries += target_data['original_query']

        for key in test_metrics_dict:
            test_metrics_dict[key] = test_metrics_dict[key] / len(dataloader)

        result_dict = {
            "model_params": self.model.model_config,
            "test_metrics": test_metrics_dict,
            "predicted_queries": predicted_queries,
            "true_queries": true_queries,
            "input_questions": input_questions
        }

        model_dir_name, model_name = self.config["inference_model_name"].split('/')
        model_name = self.config["inference_model_name"].split('/')[-1].replace(".pt", "")
        save_preds_path = os.path.join(os.environ['PROJECT_PATH'], self.config['save_model_path'],
                                       model_dir_name,
                                       f'{model_name}_predictions.json')
        utils.save_dict(result_dict, save_preds_path)
        return result_dict