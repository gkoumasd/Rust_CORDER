from tqdm import *
import os
import random

class synthetic_data():
    def __init__(self, data_path: str, flag: str, file_type='asm'):
        self.data_path = data_path
        self.flag = flag
        self.file_type = file_type
        self.file_type = '.' + self.file_type
        
    def create_sythetic_data(self):
        for subdir , dirs, files in os.walk(self.data_path):
            for file in tqdm(files):
                if file.endswith(self.file_type):
                    file_path = os.path.join(subdir, file)
                    #print(file_path)
                    
                    with open(file_path, "r", errors='ignore') as f:
                          code_snippet = str(f.read())
                          
                    if self.file_type=='.asm' and 'unsafe' in file_path: #works on high-level language
                         content_code_snipptet = code_snippet.split('\n')
                         #trimmed content without the top and btm
                         code_snipptet_trimmed = [content for j, content in enumerate(content_code_snipptet) if j!=0 and j!=len(content_code_snipptet)-2]
                         new_code_snippet = ''
                         #add on top of code snippet two random lines from trimmed code snippet.
                         new_code_snippet +=code_snipptet_trimmed[random.randrange(len(code_snipptet_trimmed))] + '\n'
                         new_code_snippet +=code_snipptet_trimmed[random.randrange(len(code_snipptet_trimmed))] + '\n'
                         #add the rest of code snippet
                         new_code_snippet +=code_snippet
                         #save the new code snippet
                         new_file =  file.split('.')
                         new_file = new_file[0] + '_synthetic.' + new_file[1]
                         save_path = os.path.join(subdir, new_file)
                         #print(save_path)
                         with open(save_path, 'w') as f:
                             f.write(new_code_snippet)
                         
                         
                         