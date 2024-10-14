import json
import os

class AtomAndCompoundCache:

    def __init__(self, parser_dict, query_key_name, kb_id_key_name, return_compound_list_flag, compound_cache_path=None):
        self.query_parser_dict = parser_dict
        self.query_to_atoms = {}
        self.query_to_compounds = {}
        self.query_key_name = query_key_name
        self.kb_id_key_name = kb_id_key_name
        self.return_compound_list = return_compound_list_flag

        first_parser = list(self.query_parser_dict.values())[0]
        self.parsers_env_list = first_parser.parser_compounds

        if compound_cache_path:
            self.load_cache(compound_cache_path)
            print("Loaded parser cache!")


    def get_atoms(self, query_sample, kb_id=None):
        query = query_sample
        if self.query_key_name is not None and self.kb_id_key_name is not None:
            query = query_sample[self.query_key_name]
            kb_id = query_sample[self.kb_id_key_name]
        if query not in self.query_to_atoms:
            atoms = self.query_parser_dict[kb_id].get_atoms(query)
            self.query_to_atoms[query] = atoms
        else:
            atoms = self.query_to_atoms[query]
        return atoms

    def extract_compounds_list(self, compound_dict):
        compound_list = []
        for compound_name in compound_dict:
            compound_list += compound_dict[compound_name]
        return compound_list

    def get_compounds(self, query_sample, kb_id=None):
        query = query_sample
        if self.query_key_name is not None and self.kb_id_key_name is not None:
            query = query_sample[self.query_key_name]
            kb_id = query_sample[self.kb_id_key_name]
        if query not in self.query_to_compounds:
            compounds = self.query_parser_dict[kb_id].get_compounds(query)
            self.query_to_compounds[query] = compounds
        else:
            compounds = self.query_to_compounds[query]
        if self.return_compound_list:
            compounds = self.extract_compounds_list(compounds)
        return compounds

    def load_cache(self, dir_path):
        self.query_to_atoms = json.load(open(os.path.join(dir_path, f'query2atom_dump.json'), 'r'))
        for key in self.query_to_atoms:
            self.query_to_atoms[key] = set(self.query_to_atoms[key])

        self.query_to_compounds = json.load(open(os.path.join(dir_path, f'query2compound_dump.json'), 'r'))
        print(f'Loaded {len(self.query_to_atoms)} from {dir_path}')

    def dump_cache(self, dir_path):
        for key in self.query_to_atoms:
            self.query_to_atoms[key] = list(self.query_to_atoms[key])
        json.dump(self.query_to_atoms, open(os.path.join(dir_path, f'query2atom_dump.json'), 'w'),
                  ensure_ascii=False, indent=4)

        json.dump(self.query_to_compounds, open(os.path.join(dir_path, f'query2compound_dump.json'), 'w'),
                  ensure_ascii=False, indent=4)
        print(f'Atom and compound dump is saved to {dir_path}')
