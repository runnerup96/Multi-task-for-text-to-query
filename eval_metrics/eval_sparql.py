import re
from eval_metrics import graph_match


class EvalSPARQL:
    def __init__(self, expected_metrics_list=[]):
        self.all_eval_functions = {'exact_match': self.eval_exact_match,
                                   'graph_match': self.eval_graph_match}

        if len(expected_metrics_list) == 0:
            expected_metrics_list = list(self.all_eval_functions.keys())
        self.used_eval_functions = {metric_name: self.all_eval_functions[metric_name]
                                    for metric_name in expected_metrics_list}


    def eval_graph_match(self, pred_query, true_query):
        true_triplet = self.get_triplet_from_sparql(true_query)
        pred_triplet = self.get_triplet_from_sparql(pred_query)
        graph1 = graph_match.Graph(true_triplet)
        graph2 = graph_match.Graph(pred_triplet)
        return graph1.get_metric(graph2)

    def get_triplet_from_sparql(self, sparql_query):
        triplet = re.findall(r"{(.*?)}", sparql_query)
        if triplet:
            triplet = triplet[0].split()
            triplet = ' '.join([elem for elem in triplet if elem]).strip()
        else:
            triplet = ''
        return triplet

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
