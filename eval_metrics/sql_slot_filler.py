

class SQLSlotFiller:
    def __init__(self):
        pass

    def check_and_fill(self):
        """
        checks if string contains substring and fills
        if does not contain - return False
        else return True and filled string
        :return:
        """

    def fill_pred_query(self, generated_query, attr_mapping_dict, db_id):
        # TODO: Add successfull filling status - if all filled - return query, else None
        request_attributes = attr_mapping_dict['request']['attr']
        comparison_data = attr_mapping_dict['comparison']
        filled_query = generated_query
        # fill request
        for old_attr, new_attr in request_attributes:
            old_attr = f'"{old_attr}"'
            filled_query = filled_query.replace(new_attr, old_attr, 1)
        # fill comparison
        attr_list, val_list, op_list = comparison_data['attr'], comparison_data['vals'], comparison_data['ops']
        for attr, op, val in zip(attr_list, op_list, val_list):
            old_attr, new_attr = attr
            old_val, new_val = val
            old_attr, old_val = f'"{old_attr}"', f'"{old_val}"'
            new_format = f"{new_attr} {op} {new_val}"
            old_format = f"{old_attr} {op} {old_val}"
            filled_query = filled_query.replace(new_format, old_format, 1)
        # all of a sudden also replace table
        filled_query = filled_query.replace('FROM table WHERE', f"FROM {db_id} WHERE")
        return filled_query

    def fill_gold_query(self, gold_query, attr_mapping_dict, db_id):
        request_attributes = attr_mapping_dict['request']['attr']
        comparison_data = attr_mapping_dict['comparison']
        filled_query = gold_query
        for old_attr, _ in request_attributes:
            old_attr_with_quotes = f'"{old_attr}"'
            filled_query = filled_query.replace(old_attr, old_attr_with_quotes, 1)
        attr_list, val_list, op_list = comparison_data['attr'], comparison_data['vals'], comparison_data['ops']
        for attr, op, val in zip(attr_list, op_list, val_list):
            old_attr, _ = attr
            old_val, _ = val
            old_attr_with_quotes, old_val_with_quotes = f'"{old_attr}"', f'"{old_val}"'
            old_format = f"{old_attr} {op} {old_val}"
            old_format_with_quotes = f"{old_attr_with_quotes} {op} {old_val_with_quotes}"
            filled_query = filled_query.replace(old_format, old_format_with_quotes, 1)
        filled_query = filled_query.replace('FROM table WHERE', f"FROM {db_id} WHERE")
        return filled_query

