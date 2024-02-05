#imports
import sys
import pandas as pd
import os
import json
import re

# global static varialbes
FILES_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace("Scripts","Data\\raw_data")
ALL_FILES = os.listdir(FILES_DIRECTORY)

CHUNK = 2 ** 10
MAX_COLWITH = 2 ** 20 - 1

# global mutatable variables
file_size = -1
start_index = 0

def get_index(bytes, substring):
    try:
        return bytes.index(substring)
    except:
        return -1

def find_next_data(f):
    global start_index

    f.seek(start_index, 0)
    bytes = f.read(CHUNK)

    end_relative_index = get_index(bytes, b'}\n')

    # if '}' was not found in chunk
    # load a new chunk and search again
    while (end_relative_index == -1):
        bytes += f.read(CHUNK)
        end_relative_index = get_index(bytes, b'}\n')
    
    data = bytes[:end_relative_index+1].decode('utf-8')
    start_index += end_relative_index + 2

    return json.loads(data)    

def prepare_data(data):
    try:
        article_id = data['article_id']
        text = ' '.join(data['article_text'])
        text = re.sub(r'(\n|\;)','', text)
        return [article_id, text]
    except Exception as e:
        print(e)
        return [None,None]

def dataframe_from_file(number, file_name = 'train.txt'):
    print(f'getting {number} articles from file: {file_name}')
    global file_size
    counter = 0

    file_path = f'{FILES_DIRECTORY}\\{file_name}'
    file_size = os.path.getsize(file_path)
    
    pd.set_option('display.max_colwidth', MAX_COLWITH)
    df = pd.DataFrame(index=range(number), columns=['article_id', 'article_text'])
    

    max_lenght = 0
    with open(file_path, 'rb') as f:
        while (counter <= number and start_index < file_size):
            data = find_next_data(f)
            data = prepare_data(data)
            
            if (data[1] != '' and data[1] != None
            and len(data[1]) >= 500
            and len(data[1]) <= MAX_COLWITH
            and '{document}' not in data[1]):
                df['article_id'][counter] = data[0]
                df['article_text'][counter] = data[1]
                counter += 1

                if len(data[1]) > max_lenght:
                    max_lenght = len(data[1])

    df = df.reset_index(drop=True)
    return df
