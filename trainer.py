import os

from tqdm import tqdm
import numpy as np

import utils


class Trainer:
    def __init__(self, model, config, evaluator, target_track_metric='exact_match', train_phase=True):
        self.model = model
        self.config = config
        self.model_config = self.config['model'][self.model.model_name]
        self.evaluator = evaluator
        self.target_track_metric = target_track_metric
        assert self.target_track_metric in self.evaluator.used_eval_functions

        self.epoch_num = self.model_config['epochs_num']

        self.save_dir_path = os.path.join(os.environ["PROJECT_PATH"], self.config['save_model_path'],
                                          self.config['run_name'])
        if train_phase:
            self.logger = utils.TXTLogger(work_dir=self.save_dir_path)
            self.model_config['model_name'] = self.model.model_name
            utils.save_dict(self.model_config, os.path.join(self.save_dir_path, 'model_config.json'))
            # log whole config to model
            self.logger.log(self.config)

    def train_with_enviroments(self, train_env_dict, val_env_dict):
        epoch = 0
        best_current_metric, current_track_metric = 0, 0
        epoch_metrics_dict = {key: 0 for key in self.evaluator.used_eval_functions}

        train_steps_per_epoch = min([len(dataloader) for dataloader in train_env_dict.values()])
        val_steps_per_epoch = min([len(dataloader) for dataloader in val_env_dict.values()])

        try:
            for epoch in tqdm(range(self.epoch_num), desc='Total epochs'):
                overall_train_epoch_loss = 0

                self.model.train()
                train_env_iter_dict = {k: iter(v) for k, v in train_env_dict.items()}
                train_per_env_dict_loss = {key: 0 for key in train_env_iter_dict}
                for _ in tqdm(range(train_steps_per_epoch), leave=False, total=train_steps_per_epoch, desc="Train"):
                    for env_name in train_env_iter_dict:
                        try:
                            batch = next(train_env_iter_dict[env_name])
                        except StopIteration:
                            # if no new batches of data, we move to new env
                            continue

                        input_data, target_data = batch['input'], batch['target']

                        train_batch_loss = self.model.train_on_batch(input_data, target_data, env_name)
                        train_per_env_dict_loss[env_name] += train_batch_loss
                        overall_train_epoch_loss += train_batch_loss

                train_per_env_dict_loss = {key: value / train_steps_per_epoch for key, value in
                                           train_per_env_dict_loss.items()}
                overall_train_epoch_step = overall_train_epoch_loss / train_steps_per_epoch

                self.model.eval()
                overall_val_epoch_loss = 0
                val_env_iter_dict = {k: iter(v) for k, v in val_env_dict.items()}
                val_per_env_dict_loss = {key: 0 for key in val_env_iter_dict}
                epoch_metrics_dict = {key: 0 for key in self.evaluator.used_eval_functions}
                for _ in tqdm(range(val_steps_per_epoch), leave=False, total=val_steps_per_epoch, desc="Validation"):
                    for env_name in val_env_iter_dict:
                        try:
                            batch = next(val_env_iter_dict[env_name])
                        except StopIteration:
                            continue

                        input_data, target_data = batch['input'], batch['target']

                        val_batch_dict = self.model.evaluate_batch(input_data, target_data, env_name)
                        val_per_env_dict_loss[env_name] += val_batch_dict['loss']
                        overall_val_epoch_loss += val_batch_dict['loss']

                        if env_name == 'full':
                            val_metrics = self.evaluator.calculate_batch_metrics(val_batch_dict['predicted_query'],
                                                                                 target_data['original_query'])
                            for key in val_metrics:
                                epoch_metrics_dict[key] += val_metrics[key]

                            last_full_sample = {
                                "original_question": input_data['original_question'][:5],
                                "original_query": target_data['original_query'][:5],
                                "predicted_query": val_batch_dict['predicted_query'][:5],
                            }

                val_per_env_dict_loss = {key: value / val_steps_per_epoch for key, value in
                                         val_per_env_dict_loss.items()}
                overall_val_epoch_step = overall_val_epoch_loss / val_steps_per_epoch

                for key in epoch_metrics_dict:
                    epoch_metrics_dict[key] = epoch_metrics_dict[key] / val_steps_per_epoch

                self.logger.log({"epoch": epoch,
                                 "train_loss": overall_train_epoch_step,
                                 "train_components_loss": train_per_env_dict_loss,
                                 "val_loss": overall_val_epoch_step,
                                 "val_components_loss": val_per_env_dict_loss,
                                 "val_metrics": epoch_metrics_dict})

                self.logger.log('********** Translation example **********')
                for input_question, true_query, pred_query in zip(last_full_sample['original_question'],
                                                                  last_full_sample['original_query'],
                                                                  last_full_sample['predicted_query']):
                    self.logger.log(f"NL: {input_question}")
                    self.logger.log(f"AQ: {true_query}")
                    self.logger.log(f"PQ: {pred_query}")
                    self.logger.log(" ")
                self.logger.log('******************************')

                current_track_metric = epoch_metrics_dict[self.target_track_metric]
                if current_track_metric > best_current_metric:
                    best_current_metric = current_track_metric
                    utils.save_model(model=self.model,
                                     optimizer_list=[self.model.t5_optimizer,
                                                     *list(self.model.env_optimizers_dict.values())],
                                     dir_path=self.save_dir_path,
                                     filename=f"epoch_{epoch}_tm_{round(best_current_metric, 2)}_{self.model.model_name}.pt")
        except KeyboardInterrupt:
            pass

        utils.save_model(model=self.model,
                         optimizer_list=[self.model.t5_optimizer, *list(self.model.env_optimizers_dict.values())],
                         dir_path=self.save_dir_path,
                         filename=f"epoch_{epoch}_tm_{round(current_track_metric, 2)}_{self.model.model_name}.pt")
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Result metrics dict: ", epoch_metrics_dict)

    def train(self, train_dataloader, val_dataloader):
        epoch = 0
        best_current_metric, current_track_metric = 0, 0
        epoch_metrics_dict = {key: 0 for key in self.evaluator.used_eval_functions}

        try:
            for epoch in tqdm(range(self.epoch_num)):
                train_epoch_loss = 0
                self.model.train()
                for batch in train_dataloader:
                    input_data, target_data = batch['input'], batch['target']
                    train_batch_loss = self.model.train_on_batch(input_data, target_data)
                    train_epoch_loss += train_batch_loss
                train_epoch_loss = train_epoch_loss / len(train_dataloader)

                val_epoch_loss = 0
                self.model.eval()
                epoch_metrics_dict = {key: 0 for key in self.evaluator.used_eval_functions}
                for batch in val_dataloader:
                    input_data, target_data = batch['input'], batch['target']

                    eval_result = self.model.evaluate_batch(input_data=input_data, target_data=target_data)
                    val_epoch_loss += eval_result['loss']

                    val_metrics = self.evaluator.calculate_batch_metrics(eval_result['predicted_query'],
                                                                         target_data['original_query'])
                    for key in val_metrics:
                        epoch_metrics_dict[key] += val_metrics[key]

                val_epoch_loss = val_epoch_loss / len(val_dataloader)
                for key in epoch_metrics_dict:
                    epoch_metrics_dict[key] = epoch_metrics_dict[key] / len(val_dataloader)

                if hasattr(self.model, 'optimizer_scheduler'):
                    learning_rate = self.model.optimizer_scheduler.optimizer.param_groups[0]['lr']
                else:
                    learning_rate = self.model.optimizer.param_groups[0]['lr']

                self.logger.log({"epoch": epoch,
                                 "train_loss": train_epoch_loss,
                                 "val_loss": val_epoch_loss,
                                 "val_metrics": epoch_metrics_dict,
                                 "learning_rate": learning_rate})

                self.logger.log('********** Translation example **********')
                for input_question, true_query, pred_query in zip(input_data['original_question'][:5],
                                                                  target_data['original_query'][:5],
                                                                  eval_result['predicted_query'][:5]):
                    self.logger.log(f"NL: {input_question}")
                    self.logger.log(f"AQ: {true_query}")
                    self.logger.log(f"PQ: {pred_query}")
                    self.logger.log(" ")
                self.logger.log('******************************')

                current_track_metric = epoch_metrics_dict[self.target_track_metric]
                if current_track_metric > best_current_metric:
                    best_current_metric = current_track_metric
                    utils.save_model(model=self.model,
                                     optimizer_list=[self.model.optimizer],
                                     dir_path=self.save_dir_path,
                                     filename=f"epoch_{epoch}_tm_{round(best_current_metric, 2)}_{self.model.model_name}.pt")
        except KeyboardInterrupt:
            pass

        utils.save_model(model=self.model,
                         optimizer_list=[self.model.optimizer],
                         dir_path=self.save_dir_path,
                         filename=f"epoch_{epoch}_tm_{round(current_track_metric, 2)}_{self.model.model_name}.pt")
        print(f'Dump model to {self.config["save_model_path"]} on {epoch} epoch!')
        print("Result metrics dict: ", epoch_metrics_dict)
