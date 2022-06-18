# todo: write evaluation utils:
"""
1. check number of output topics
2. get topic sizes
3. get_info about a topic: topic_words, word_scores, topic_document_indices = model.get_topics(77) <from top2vec github page>
4. search topics with a keyword: topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["medicine"], num_topics=5)
5. generate word clouds:
topic_words, word_scores, topic_scores, topic_nums = model.search_topics(keywords=["medicine"], num_topics=5)
for topic in topic_nums:
    model.generate_topic_wordcloud(topic)
6. search documents by topic: documents, document_scores, document_ids = model.search_documents_by_topic(topic_num=48, num_docs=5)
"""
import numpy as np
import pandas as pd
import umap.umap_ as umap
from scipy.optimize import linear_sum_assignment as linear_assignment
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
import itertools
import numpy as np
from typing import List
import pandas as pd

import plotly.graph_objects as go
from plotly.subplots import make_subplots
from config import WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN
import seaborn as sns

sns.set()


def c_tf_idf(documents, m, ngram_range=(1, 1)):
    count = CountVectorizer(ngram_range=ngram_range, stop_words="english").fit(documents)
    t = count.transform(documents).toarray()
    w = t.sum(axis=1)
    tf = np.divide(t.T, w)
    sum_t = t.sum(axis=0)
    idf = np.log(np.divide(m, sum_t)).reshape(-1, 1)
    tf_idf = np.multiply(tf, idf)
    return tf_idf, count


def extract_top_n_words_per_topic(tf_idf, count, docs_per_topic, n=20):
    words = count.get_feature_names()
    labels = list(docs_per_topic.Topic)
    tf_idf_transposed = tf_idf.T
    indices = tf_idf_transposed.argsort()[:, -n:]
    top_n_words = {label: [(words[j], tf_idf_transposed[i][j]) for j in indices[i]][::-1] for i, label in
                   enumerate(labels)}
    return top_n_words


def get_top_n_words_in_cluster(top_n_words, n_words_to_check=20, cluster_to_check=0):
    return top_n_words[cluster_to_check][:n_words_to_check]


def extract_topic_sizes(df):
    topic_sizes = (df.groupby(['Topic'])[WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN]
                   .count()
                   .reset_index()
                   .rename({"Topic": "Topic", WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN: "Size"}, axis='columns')
                   .sort_values("Size", ascending=False))
    return topic_sizes


def plot_in_2d(embeddings, clusters):
    # Prepare data
    umap_2d_embedding = umap.UMAP(n_neighbors=15, n_components=2, min_dist=0.0, metric='cosine').fit_transform(
        embeddings)
    result = pd.DataFrame(umap_2d_embedding, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    # Visualize clusters
    fig, ax = plt.subplots(figsize=(20, 10))
    outliers = result.loc[result.labels == -1, :]
    clustered = result.loc[result.labels != -1, :]
    plt.scatter(outliers.x, outliers.y, color='#BDBDBD', s=0.05)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=0.05, cmap='hsv_r')
    plt.colorbar()


def plot_confusion_metrics_and_print_class_mapping(labels, predicted_clusters):
    classes_to_cluster_labels_random_mapping = dict(zip(set(labels), set(predicted_clusters)))
    # print(classes_to_cluster_labels_wrong_mapping)
    # cluster_pseudo_labels = [cluster_labels_to_classes_wrong_mapping[cluster_original_label] for cluster_original_label in list(clusters.labels_)]
    confusion_metrics_labels_as_numbers = [classes_to_cluster_labels_random_mapping[class_original_label] for
                                           class_original_label in labels]
    cm = confusion_matrix(confusion_metrics_labels_as_numbers,
                          predicted_clusters)  # labels=list(classes_to_cluster_labels_wrong_mapping.keys()))

    def _make_cost_m(cm):
        s = np.max(cm)
        return (- cm + s)

    indexes = linear_assignment(_make_cost_m(cm))
    old_to_new_predicted_cluster_mapping = {i: j for i, j in zip(indexes[0], indexes[1])}
    # classes_to_cluster_labels_right_mapping = {class_label: old_to_new_predicted_cluster_mapping[cluster_label] for
    #                                            class_label, cluster_label in
    #                                            classes_to_cluster_labels_random_mapping.items()}
    cm[:, list(old_to_new_predicted_cluster_mapping.keys())] = cm[:,
                                                               list(old_to_new_predicted_cluster_mapping.values())]
    ax = sns.heatmap(cm, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
    accuracy = np.trace(cm) / np.sum(cm)
    print(
        f"class label mapping: {classes_to_cluster_labels_random_mapping} \n\ncluster indexes change: {old_to_new_predicted_cluster_mapping} \n\naccuracy: {accuracy}")


def visualize_top_words_in_different_clusters(topic_word_freq):
    colors = itertools.cycle(["#D55E00", "#0072B2", "#CC79A7", "#E69F00", "#56B4E9", "#009E73", "#F0E442"])
    width = 250
    height = 250

    # Initialize figure
    subplot_titles = [f"Topic {topic}" for topic in topic_word_freq]
    columns = 4
    rows = int(np.ceil(len(topic_word_freq) / columns))
    fig = make_subplots(rows=rows,
                        cols=columns,
                        shared_xaxes=False,
                        horizontal_spacing=.1,
                        vertical_spacing=.4 / rows if rows > 1 else 0,
                        subplot_titles=subplot_titles)

    # Add barchart for each topic
    row = 1
    column = 1
    for topic in topic_word_freq:
        words = [tup[0] for tup in topic_word_freq[topic]]
        scores = [tup[1] for tup in topic_word_freq[topic]]

        fig.add_trace(
            go.Bar(x=scores,
                   y=words,
                   orientation='h',
                   marker_color=next(colors)),
            row=row, col=column)

        if column == columns:
            column = 1
            row += 1
        else:
            column += 1

    # Stylize graph
    fig.update_layout(
        template="plotly_white",
        showlegend=False,
        title={
            'text': "<b>Topic Word Scores",
            'x': .5,
            'xanchor': 'center',
            'yanchor': 'top',
            'font': dict(
                size=22,
                color="Black")
        },
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"
        ),
    )

    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)

    return fig
