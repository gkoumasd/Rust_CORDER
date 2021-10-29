from os import path

from tree_sitter import Language, Parser
from pathlib import Path
import glob, os
import logging
import platform
import tree_sitter_parsers

class ASTParser():
    LOGGER = logging.getLogger('ASTParser')
    def __init__(self, language='rust'):
        # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        plat = platform.system()     
        p = path.join(home, ".tree-sitter")
        if os.path.exists(path.join(p, "tree-sitter-parsers-" + plat)):
            os.chdir(path.join(p, "tree-sitter-parsers-" + plat))
        else:
            os.chdir(path.join(p, "bin"))
        self.Languages = {}
        for file in glob.glob("*.so"):
          try:
            lang = os.path.splitext(file)[0]
            if os.path.exists(path.join(p, "tree-sitter-parsers-" + plat)):
                self.Languages[lang] = Language(path.join(p, "tree-sitter-parsers-" + plat, file), lang)
            else:
                self.Languages[lang] = Language(path.join(p, "bin", file), lang)
          except:
            print("An exception occurred to {}".format(lang))
        os.chdir(cd)
        self.parser = Parser()
        
        self.language = language
        if self.language == None:
            self.LOGGER.info("Cannot find language configuration, using rust parser as the default to parse the code into AST")
            self.language = "rust"

        lang = self.Languages.get(self.language)
        self.parser.set_language(lang)
        # -----------------------------------------------------------------

       
    def parse_with_language(self, code_snippet, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)
        return self.parser.parse(code_snippet)

    def parse(self, code_snippet):
        return self.parser.parse(code_snippet)
    
    def set_language(self, language):
        lang = self.Languages.get(language)
        self.parser.set_language(lang)