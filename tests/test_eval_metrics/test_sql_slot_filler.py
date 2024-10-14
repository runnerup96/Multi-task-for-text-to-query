import unittest
from eval_metrics import sql_slot_filler


class TestSQLSlotFiller(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        cls.sql_slot_filler = sql_slot_filler.SQLSlotFiller()

    def test_fill_correct_pred_query(self):
        generated_query = "SELECT Notes_str FROM table WHERE Current_slogan = STR_VALUE_1"
        attr_mapping_dict = {
            "request": {
                "attr": [
                    [
                        "Notes str",
                        "Notes_str"
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
        expected_query = 'SELECT "Notes str" FROM table_1 WHERE "Current slogan" = "SOUTH AUSTRALIA"'
        filled_query = self.sql_slot_filler.fill_pred_query(generated_query, attr_mapping_dict, 1)
        self.assertEqual(expected_query, filled_query)


    def test_fill_correct_gold_query(self):
        pass

    def test_fill_incorrect_pred_query(self):
        generated_query = "SELECT Player FROM table WHERE Position = STR_VALUE_1 AND Years_in_Toronto = STR_VALUE_2"
        attr_mapping_dict = {
            "request": {
                "attr": [
                    [
                        "School/Club Team",
                        "School/Club_Team"
                    ]
                ]
            },
            "comparison": {
                "attr": [
                    [
                        "No.",
                        "No."
                    ]
                ],
                "vals": [
                    [
                        "21",
                        "NUM_VALUE_1"
                    ]
                ],
                "ops": [
                    "="
                ]
            }
        }

        expected_query = 'SELECT "Notes str" FROM table_1 WHERE "Current slogan" = "SOUTH AUSTRALIA"'
        filled_query = self.sql_slot_filler.fill_pred_query(generated_query, attr_mapping_dict, 1)
        self.assertEqual(expected_query, filled_query)
