#imports
import sys
import os
import numpy as np
import pandas as pd
import re
import converter_xlsx as converter
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import PorterStemmer
import gensim
import gensim.corpora as corpora
from gensim.models import CoherenceModel
from ast import literal_eval

# Control variables
VERSION = 4
DATA_AMOUNT = 500

REPEATS = 20
PASSES = 20
EXPLORATORY_STEP = 5

# Global static variables
FILE_EXTENTION = 'csv'

BASE_DIRECTORY = os.path.dirname(os.path.realpath(__file__))
CLEANED_DATA_DIRECTORY = BASE_DIRECTORY.replace("Scripts", "Data\\cleaned_data")
MODELS_INFO_DIRECTORY = BASE_DIRECTORY.replace("Scripts", "Data\\models_info")

def get_cleaned_df(data_amount = DATA_AMOUNT):
    file_name = f'{data_amount}.{FILE_EXTENTION}'

    if file_name in os.listdir(CLEANED_DATA_DIRECTORY):
        print(f'opening existing cleaned file: {file_name}')
        df = pd.read_csv(f'{CLEANED_DATA_DIRECTORY}\\{file_name}', sep=';')['article_text']
        df = df.apply(literal_eval) # convert the string into a list of strings
        return df
    else:
        print(f'creating new cleaned file: {file_name}')
        df = converter.dataframe_from_file(data_amount)['article_text']
        stop_words = stopwords.words('english')
        stop_words.extend(['et','al','one','two'])

        def prepare_series(data):
            data = data.lower()
            data = re.sub('[^a-z\s]+', '', data)
            data = word_tokenize(data)
            data = [word for word in data if word not in stop_words and len(word)>1]
            return data
        print('cleaning file')
        df = df.map(prepare_series)
        print('saving file')
        df.to_csv(f'{CLEANED_DATA_DIRECTORY}\\{file_name}', sep=';')
        return df

def open_models_database():
    # verificar se arquivo existe
    name = f'{DATA_AMOUNT}_v{VERSION}'
    file_name = f'database_{name}.xlsx'
    new_folder = f'{MODELS_INFO_DIRECTORY}\\{name}'

    # criar pasta de organizacao dos modelos prontos caso não exista
    if not os.path.exists(new_folder):
        os.makedirs(new_folder)
        
    if file_name in os.listdir(new_folder):
        print('opening existing database file')
        df = pd.read_excel(f'{new_folder}\\{file_name}')
    else:
        print('creating new database file')
        basic_dict = {
            'topic_amount':[1, 5, 10],
            'perplexity': [np.NaN, np.NaN, np.NaN],
            'RPC': [np.NaN, np.NaN, np.NaN]
        }
        df = pd.DataFrame.from_dict(basic_dict)

        df.to_excel(f'{new_folder}\\{file_name}', index=False)
    print(df)
    return df    

def topics_based_on(df_input):
    df = df_input.copy()

    valor = df['RPC'].idxmin()
    ascending_rule = True

    df_aux = df.iloc[valor - 1:valor + 2].\
                sort_values(by = 'RPC', ascending = ascending_rule).\
                iloc[:-1].\
                reset_index(drop=True)
    
    if abs(df_aux.at[0, 'topic_amount'] - df_aux.at[1, 'topic_amount']) == 1:
        return -1
    else:
        return int(((df_aux.at[1, 'topic_amount'] - df_aux.at[0, 'topic_amount'])/2) +\
                    df_aux.at[0, 'topic_amount'])

def choose_number_of_topics(df_input):
    df = df_input.copy()

    df_filtered = df.loc[df['perplexity'].isna()]
    if df_filtered.shape[0] > 0:
        return df_filtered.iloc[0,0]
    elif df['RPC'].min() == df['RPC'].iloc[-1]:
        return df['topic_amount'].max() + EXPLORATORY_STEP
    else:
        return topics_based_on(df)

def prepare_for_lda(df_input):
    print("preparing for LDA")
    df = df_input.copy()

    print("\tapplying Porters Stemmer")
    df = df.apply(lambda x: [PorterStemmer().stem(word) for word in x])
    
    # deal with common and rare words here

    print("\tcreating id2word and corpus")
    id2word = corpora.Dictionary(df)

    corpus = df.map(lambda x: id2word.doc2bow(x))

    return df, id2word, corpus

def lda_folder_location(num_topics):
    partial_directory = f'{DATA_AMOUNT}_v{VERSION}\\topics_{num_topics}'

    full_directory = f'{MODELS_INFO_DIRECTORY}\\{partial_directory}'

    if not os.path.exists(full_directory):
        os.makedirs(full_directory)

    return full_directory

def calculate_lda(id2word,
                  corpus,
                  num_topics,
                  eval_every=1,
                  ):
    
    perplexity_values = []
    print(f'calculating model with number of topics: {num_topics}')
    for i in range(REPEATS):
        print(f'\titeration: {i}')
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                        id2word=id2word,
                                        num_topics=num_topics,
                                        alpha='auto',
                                        eta='auto',
                                        eval_every=eval_every,
                                        passes=PASSES
                                        )
         
        bound = lda_model.log_perplexity(corpus)
        perplexity_value = np.exp2(-bound)
        print(f'\tPerplexity: {perplexity_value}\n')
        perplexity_values.append(perplexity_value)

    return lda_model, (sum(perplexity_values) / len(perplexity_values))

def calcualte_coherence_score(lda_model):
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df, dictionary=id2word, coherence='c_v')
    return coherence_model_lda.get_coherence()

def update_RPC(df_input):
    print('updating RPC')
    df = df_input.copy()
    df_filtered = df.loc[df['perplexity'].isna()]
    if df_filtered.shape[0] == 0:
        for i in range(1, df.shape[0]):
            df.at[i, 'RPC'] = abs((df.at[i, 'perplexity'] - df.at[i - 1, 'perplexity']) /\
                                (df.at[i, 'topic_amount'] - df.at[i - 1, 'topic_amount']))
    return df


def add_new_value(df_input, new_data):
    df = df_input.copy()

    df_filter = df.loc[df['topic_amount'] == new_data[0]]

    if len(df_filter) > 0:
        index = df_filter.index[0]
        df.at[index, 'perplexity'] = new_data[1]
        return df

    new_data_df = pd.DataFrame({
        'topic_amount': new_data[0],
        'perplexity': new_data[1]
    }, index = [-1])
    
    first_half = df.loc[df['topic_amount'] < new_data[0]]
    second_half = df.loc[df['topic_amount'] > new_data[0]]

    return pd.concat([first_half, new_data_df, second_half]).reset_index(drop=True)

def save_database(df_input):
    df = df_input.copy()

    name = f'{DATA_AMOUNT}_v{VERSION}'
    file_name = f'database_{name}.xlsx'
    new_folder = f'{MODELS_INFO_DIRECTORY}\\{name}'

    df.to_excel(f'{new_folder}\\{file_name}', index=False)

def retrieve_best_num_topics():
    models_database = open_models_database()

    num_topics = -1
    for index, row in models_database.iterrows():
        if models_database.at[index, 'RPC'] < models_database.at[index + 1, 'RPC']:
            return int(row['topic_amount'])
        
def calculate_most_coherent(id2word, corpus, num_topics):
    print('calculating iteration: 0')
    best_lda = gensim.models.LdaModel(corpus=corpus,
                                id2word=id2word,
                                num_topics=num_topics,
                                alpha='auto',
                                eta='auto',
                                eval_every=PASSES,
                                passes=20
                                )
    best_coherence = calcualte_coherence_score(best_lda)
    print(f'\tcoherence score: {best_coherence}')

    for i in range(REPEATS):
        print(f'calculating iteration: {i+1}')
        lda_model = gensim.models.LdaModel(corpus=corpus,
                                    id2word=id2word,
                                    num_topics=num_topics,
                                    alpha='auto',
                                    eta='auto',
                                    eval_every=1,
                                    passes=PASSES
                                    )
        coherence = calcualte_coherence_score(lda_model)
        print(f'\tcoherence score: {coherence}')
        if coherence > best_coherence:
            best_coherence = coherence
            best_lda = lda_model
    
    return best_lda, best_coherence


if __name__ == '__main__':
    df = get_cleaned_df()
    df, id2word, corpus = prepare_for_lda(df)
    while True:
        print('starting iteration')
        models_database = open_models_database()
        number = choose_number_of_topics(models_database)
        if number == -1:
            print('no more calculations')
            break
        _, per = calculate_lda(id2word, corpus, number)
        models_database = add_new_value(models_database, [number, per])
        models_database = update_RPC(models_database)
        save_database(models_database)
        print('\n\n')
    
    #retrieve best number
    num_topics = retrieve_best_num_topics()
    
    #calculating best coherence
    best_lda, best_coherence = calculate_most_coherent(id2word, corpus, num_topics)
    
    name = f'{DATA_AMOUNT}_v{VERSION}'
    new_folder = f'{MODELS_INFO_DIRECTORY}\\{name}'

    best_lda.save(f'{new_folder}\\best_model')