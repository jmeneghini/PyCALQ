import yaml
import json
import xmltodict
import os
import logging

# YAML include functionality using SafeLoader
class IncludeLoader(yaml.SafeLoader):
    """Custom YAML loader that supports !include directive"""
    def __init__(self, stream):
        try:
            self._root = os.path.dirname(stream.name)
        except AttributeError:
            self._root = os.path.curdir
        super().__init__(stream)

def include_constructor(loader, node):
    """Include file referenced at node."""
    filename = loader.construct_scalar(node)
    filepath = os.path.join(loader._root, filename)

    if not os.path.exists(filepath):
        raise FileNotFoundError(f"Include file not found: {filepath}")

    with open(filepath, 'r') as f:
        return yaml.load(f, IncludeLoader)

# Add the include constructor to our custom loader
IncludeLoader.add_constructor('!include', include_constructor)

class ConfigHandler:

    def __init__( self, input ):
        self.configs = {}
        self.init( input )

    def init( self, input ):
        if type(input)==str:
            yaml_file = False; json_file = False; xml_file = False

            #try to input as yaml file
            try:
                with open(input, 'r') as f:
                    config_info = yaml.load(f, IncludeLoader)
                    yaml_file = True
            except Exception as e:
                logging.warning(f"{input} is not a YAML file: {str(e)}")
                
            #try to input as json file
            if not yaml_file: #tested
                try:
                    with open(input, 'r') as f:
                        config_info = json.load(f)
                        json_file = True
                except:
                    logging.warning(f"{input} is not a JSON file")
                    
            #try to input as xml file #problems
            if not yaml_file and not json_file: #tested
                try:
                    with open(input, 'r') as f:
                        config_info = xmltodict.parse(f.read())
                        xml_file = True
                        
                    #first level guaranteed to only have one index
                    config_info[list(config_info.keys())[0]] = dict(config_info[list(config_info.keys())[0]])
                    
                    #issue with xml everything that is input is interpreted as a string. need to do more input processing if taken seriously
                    
                except:
                    logging.warning(f"{input} is not a XML file")
                    
            if not yaml_file and not json_file and not xml_file:
                logging.error("Configs cannot be handled at this time.")
                
            #extract task name (root) and organize config info accordingly
            root = list(config_info.keys())[0]
            if type(config_info[root])==list:
                if root not in self.configs:
                    self.configs[root] = []
                self.configs[root]+=config_info[root]
            else:
                self.configs.update(config_info)
                
        elif type(input)==dict:
            self.configs.update(input)
            
        elif type(input)==list:
            for this_input in input:
                self.init(this_input )
        else:
            logging.error("Configs cannot be handled at this time.")

    #perform basic checks as well as check for requirements
    def check(self, root, requirements ):

        #check that root is singular and that is the set one
        if len(self.configs.keys())>1:
            logging.error(f"Config cannot have more than one root. Only '{root}' is allowed.")

        if list(self.configs.keys())[0]!=root:
            logging.error(f"{root} config must have '{root}' as the root.")

        #check that required input keys are present
        for requirement in requirements:
            if type(requirement)!=dict:
                if requirement not in self.configs[root].keys():
                    logging.error(f"Missing parameter '{requirement}' in {root} config.")
            else:
                requirement_key = list(requirement.keys())[0]
                if requirement_key not in self.configs[root].keys():
                    logging.error(f"Missing parameter '{requirement_key}' in {root} config.")
                for requirementee in requirement[requirement_key]:
                    if requirementee not in self.configs[root][requirement_key].keys():
                        logging.error(f"Missing parameter '{requirementee}' in {root} config.")


