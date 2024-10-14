from eval_metrics import sql_slot_filler, sql_executor


class EvalSQL:
    def __init__(self, table_name2db_path_dict, expected_metrics_list=[]):
        self.sql_slot_filler = sql_slot_filler.SQLSlotFiller()
        self.sql_executor = sql_executor.SQLExecutor(db_mapping_dict=table_name2db_path_dict)

        self.all_eval_functions = {'exact_match': self.eval_exact_match}

        if len(expected_metrics_list) == 0:
            expected_metrics_list = list(self.all_eval_functions.keys())

        self.used_eval_functions = {metric_name: self.all_eval_functions[metric_name]
                                    for metric_name in expected_metrics_list}

    def eval_exec_match(self, true_query, generated_query, correct_attr_mapping, db_id):
        # TODO: Finish execution match
        pred_result = None
        filled_gold_query = self.sql_slot_filler.fill_gold_query(gold_query=true_query,
                                                                 attr_mapping_dict=correct_attr_mapping,
                                                                 db_id=db_id)
        filled_generated_query = self.sql_slot_filler.fill_pred_query(generated_query=generated_query,
                                                                 attr_mapping_dict=correct_attr_mapping,
                                                                 db_id=db_id)
        true_result = self.sql_executor.execute_sql(filled_gold_query, db_id)
        if filled_generated_query:
            pred_result = self.sql_executor.execute_sql(filled_generated_query, db_id)
        result = self.compare_exec_results(true_result, pred_result)
        return result

    def compare_exec_results(self, true_res, pred_resr):
        result = 0
        if len(true_res) == 0 and len(pred_resr) == 0:
            result = 1
        else:
            true_res = set([sample[0] for sample in true_res])
            pred_res = set([sample[0] for sample in pred_resr])
            result_intersection = true_res.intersection(pred_res)
            if len(result_intersection) == len(true_res):
                result = 1
        return result

    def eval_exact_match(self, pred_query, true_query):
        pred_query = pred_query.lower()
        true_query = true_query.lower()
        exact_match_status = 0
        if pred_query == true_query:
            exact_match_status = 1
        return exact_match_status

    def calculate_batch_metrics(self, pred_query_list, true_query_list):
        result_dict = {key: 0 for key in self.used_eval_functions}
        for idx in range(len(true_query_list)):
            for eval_func_name, eval_func in self.used_eval_functions.items():
                result_dict[eval_func_name] += eval_func(pred_query_list[idx], true_query_list[idx])

        for key in result_dict:
            result_dict[key] /= len(true_query_list)

        return result_dict
