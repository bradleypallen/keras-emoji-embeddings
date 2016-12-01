import pandas as pd
import matplotlib.pyplot as plt

EMOJI_EMB_VIZ_FILE = 'emoji_emb_viz.csv'

def __visualize():
    df = pd.read_csv(EMOJI_EMB_VIZ_FILE)
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.scatter(df['x'].values, df['y'].values, marker='o', alpha=0.0)
    for k, v in df.iterrows():
        ax.annotate(v['emoji'], [ v['x'], v['y'] ])
    plt.title('t-SNE Visualization of Unicode Emoji Embeddings')
    plt.grid()
    plt.show()

if __name__ == '__main__':
    __visualize()
