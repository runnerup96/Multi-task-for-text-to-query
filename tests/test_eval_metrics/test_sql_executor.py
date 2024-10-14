import unittest
from eval_metrics import sql_executor


class TestSQLExecutor(unittest.TestCase):

    @classmethod
    def setUpClass(cls):
        name2path_dict = {
            "table_10015132_16": "dataset/wikisql/wikisql_dbs/my_dbs/table_10015132_16.sqlite3",
        }
        cls.sql_executor = sql_executor.SQLExecutor(name2path_dict)

    def test_execute_query(self):
        sql = 'SELECT "Nationality" FROM table_10015132_16 WHERE "Player" = "Terrence Ross"'
        db_name = 'table_10015132_16'
        expected_result = [('United States',)]
        result = self.sql_executor.execute_sql(sql, db_name)
        self.assertListEqual(result, expected_result)