import os
import sqlalchemy


class SQLExecutor:
    def __init__(self, db_mapping_dict):
        self.kb_mapping_dict = db_mapping_dict

    def execute_sql(self, sql, db_name):
        db_path = os.path.join(os.environ['PROJECT_PATH'], self.kb_mapping_dict[db_name])

        engine = sqlalchemy.create_engine('sqlite:///{}'.format(db_path), echo=False)

        with engine.connect() as conn:
            sql_result = (conn.execute(sqlalchemy.text(sql)).fetchall())

        return sql_result