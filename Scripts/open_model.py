#imports
import os
import gensim
import pandas as pd
import sys
from wordcloud import WordCloud
import matplotlib.pyplot as plt
import train_model
from gensim.models import CoherenceModel

# global static variables
MODELS_INFO_DIRECTORY = os.path.dirname(os.path.realpath(__file__)).replace("Scripts", "Data\\models_info")

# global variables
current_version = -1
current_data_amount = -1


def get_wordclouds(lda_model, num_rows = 4, num_cols = 3, max_words = 50):   
    wordclouds = []
    for i in range(num_rows * num_cols):
        wordcloud_obj = WordCloud(
            width=1800,
            height=700,
            background_color="white",
            prefer_horizontal=1,
            font_step=1
        )

        wordclouds.append(
            wordcloud_obj.fit_words(dict(lda_model.show_topic(i, max_words)))
        )

    _, axes = plt.subplots(nrows=num_rows, ncols=num_cols)
    index = 0
    for i in range(num_rows):
        for j in range(num_cols):
            ax = axes[i,j]

            ax.imshow(wordclouds[index])

            ax.set_title(f'topic number: {index + 1}')

            index += 1
            ax.axis("off")
    

def load_model():
    print('loading model')
    global current_version
    global current_data_amount

    print("folders:")
    for idx, folder in enumerate(os.listdir(MODELS_INFO_DIRECTORY)):
        print(f'{idx} - {folder}')
    
    folder_id = int(input('Choose folder number: '))

    folder_name = os.listdir(MODELS_INFO_DIRECTORY)[folder_id]
    print(folder_name)

    current_data_amount = folder_name.split('_v')[0]
    current_version = folder_name.split('_v')[1]

    model_folder = f'{MODELS_INFO_DIRECTORY}\\{folder_name}'

    return gensim.models.LdaModel.load(f'{model_folder}\\best_model')

def calculate_coherence(lda_model):
    print("calculating coherence score")
    train_model.VERSION = current_version
    train_model.DATA_AMOUNT = current_data_amount

    df = train_model.get_cleaned_df()
    df, id2word, corpus = train_model.prepare_for_lda(df)
    coherence_model_lda = CoherenceModel(model=lda_model, texts=df, dictionary=id2word, coherence='c_v')

    coherence_score = coherence_model_lda.get_coherence()
    print(f'coherence_score: {coherence_score}')
    return coherence_score

def plot_table(df_input):
    df = df_input.copy()

    fig, ax = plt.subplots()

    fig.patch.set_visible(False)
    ax.axis('off')
    ax.axis('tight')

    ax.table(cellText=df.values, colLabels=df.columns, loc='center')

    fig.tight_layout()


def get_result_table():
    folder_name = f'{current_data_amount}_v{current_version}'
    file_name = f'database_{current_data_amount}_v{current_version}.xlsx'

    return pd.read_excel(f'{MODELS_INFO_DIRECTORY}\\{folder_name}\\{file_name}')

def get_best_table(df_input, coherence_score = '0'):
    df = df_input.copy()

    result_df = df.nsmallest(1, 'RPC').reset_index(drop=True)
    result_df['coherence'] = coherence_score
    result_df['topic_amount'] = result_df['topic_amount'].astype(int)
    return result_df

def plot_topic_info_tables(df_input):
    df = df_input.copy()
    _, axes = plt.subplots(nrows=1, ncols=2)

    axes[0].scatter(x=df['topic_amount'], y= df['perplexity'])
    axes[0].set_xlabel('topic_amount')
    axes[0].set_ylabel('perplexity')

    axes[1].scatter(x=df['topic_amount'], y= df['RPC'])
    axes[1].set_xlabel('topic_amount')
    axes[1].set_ylabel('RPC')


if __name__ == '__main__':
    model = load_model()
    
    coherence_score = calculate_coherence(model)
    result_df = get_result_table()
    best_topic_df = get_best_table(result_df, coherence_score)

    plot_table(best_topic_df)
    plot_table(result_df)

    plot_topic_info_tables(result_df)

    get_wordclouds(model)
    plt.show()