from pathlib import Path
from os import path
import glob, os
from tree_sitter import Language, Parser
from tqdm import tqdm
import urllib.request
from urllib3.exceptions import InsecureRequestWarning

import zipfile
import shutil

import ssl
ssl._create_default_https_context = ssl._create_unverified_context

class DownloadProgressBar(tqdm):
    def update_to(self, b=1, bsize=1, tsize=None):
        if tsize is not None:
            self.total = tsize
        self.update(b * bsize - self.n)


def download_url(url, output_path):
    with DownloadProgressBar(unit='B', unit_scale=True,
                             miniters=1, desc=url.split('/')[-1]) as t:
        if (not os.environ.get('PYTHONHTTPSVERIFY', '') and getattr(ssl, '_create_unverified_context', None)):
           ssl._create_default_https_context = ssl._create_unverified_context
        urllib.request.urlretrieve(url, filename=output_path, reporthook=t.update_to)

class ASTParser():
    import logging
    LOGGER = logging.getLogger('ASTParser')
    
    def __init__(self, language='rust'):
         # ------------ To initialize for the treesitter parser ------------
        home = str(Path.home())
        cd = os.getcwd()
        
        p = path.join(home, ".tree-sitter", "bin")
        os.chdir(p)   
        print(p)
        
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
