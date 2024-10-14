import unittest
from eval_metrics import eval_sql


class TestEvalSQL(unittest.TestCase):
    #TODO: Unfinished Work
    @classmethod
    def setUpClass(cls):
        name2path_dict = {
            "table_10015132_16": "my_dbs/table_10015132_16.sqlite3",
        }
        cls.sql_eval = eval_sql.EvalSQL(table_name2db_path_dict=name2path_dict)

    def test_eval_query(self):
        db_id = "table_10015132_16"
        generated_query = "SELECT Notes FROM table WHERE Current_slogan = STR_VALUE_1"
        gold_query = "SELECT Notes FROM table WHERE Current slogan = SOUTH AUSTRALIA"
        attribute_mapping = {
            "request": {
                "attr": [
                    [
                        "Notes",
                        "Notes"
                    ]
                ]
            },
            "comparison": {
                "attr": [
                    [
                        "Current slogan",
                        "Current_slogan"
                    ]
                ],
                "vals": [
                    [
                        "SOUTH AUSTRALIA",
                        "STR_VALUE_1"
                    ]
                ],
                "ops": [
                    "="
                ]
            }
        }
        expected_result = 1
        result = self.sql_eval.eval_exec_match(true_query=gold_query, generated_query=generated_query,
                                               correct_attr_mapping=attribute_mapping, db_id=db_id)
        self.assertEqual(result, expected_result)


