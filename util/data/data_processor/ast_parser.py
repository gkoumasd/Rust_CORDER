from pathlib import Path
from os import path
import glob, os
from tree_sitter import Language, Parser

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    
    def __init__(self, language='rust'):
         # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        
        p = path.join(home, ".tree-sitter")
        
        if not path.exists(p):
            print('You have to upload the parsers')
        
        p = path.join(p, "bin")
        os.chdir(p)   
        
        langs = []
        for file in glob.glob("tree-sitter-*"):        
            lang = file.split("-")[2]
            if not "." in file.split("-")[3]: # c-sharp => c_sharp.so
                lang = lang + "_" + file.split("-")[3]
            langs.append(file)
            Language.build_library(
                # Store the library in the `build` directory
                lang + '.so',
                # Include one or more languages
                langs
            )
            
        self.Languages = {}   
        
        for file in glob.glob("*.so"):
          try:
            lang = os.path.splitext(file)[0]
            self.Languages[lang] = Language(path.join(p, file), lang)
          except:
            print("An exception occurred to {}".format(lang))
            
        os.chdir(cd)   
        
        self.parser = Parser()
        
        self.language = language
        
        lang = self.Languages.get(self.language)
        
        self.parser.set_language(lang)
        
    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)    