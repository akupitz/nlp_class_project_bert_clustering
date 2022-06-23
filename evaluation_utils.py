import itertools
import umap.umap_ as umap
import sklearn
import numpy as np
import pandas as pd
import plotly.graph_objects as go
import seaborn as sns
import hdbscan
from scipy.optimize import linear_sum_assignment as linear_assignment
from matplotlib import pyplot as plt
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.metrics import confusion_matrix
from plotly.subplots import make_subplots
from tqdm import tqdm
from functools import partial
from tqdm import trange
from sklearn.metrics.cluster import normalized_mutual_info_score
from sklearn.metrics.cluster import adjusted_rand_score
from hyperopt import fmin, tpe, hp, STATUS_OK, space_eval, Trials
from config import WHY_NOT_ETHICAL_CLEAN_TEXT_COLUMN, RISK_1_COLUMN

sns.set()


def random_search(embeddings, space, num_evals):
    '''
    Random search of HDBSCAN hyperparameter spcae
    Arguments:
        embeddings: embeddings to use
        space: dict, hyperparameter space to search with keys of
               'min_cluster_size' and 'min_samples' and values as ranges
        num_evals: int, number of trials to run
    Returns:
        result_df: dataframe of run_id, min_cluster_size, min_samples,
                   total number of clusters, and percent of data labeled as noise

    '''

    results = []

    for i in range(num_evals):
        min_cluster_size = int(np.random.choice(space['min_cluster_size']))
        min_samples = int(np.random.choice(space['min_samples']))

        # print(f"min cluster size: {min_cluster_size}, min samples: {min_samples}")
        clusters_hdbscan = (hdbscan.HDBSCAN(min_cluster_size=min_cluster_size,
                                            min_samples=min_samples,
                                            metric='euclidean',
                                            cluster_selection_method='eom')
                            .fit(embeddings))

        labels = clusters_hdbscan.labels_
        label_count = len(np.unique(labels))

        total_num = len(clusters_hdbscan.labels_)
        cost = (np.count_nonzero(clusters_hdbscan.probabilities_ < 0.05) / total_num)

        results.append([i, min_cluster_size, min_samples, label_count, cost])

    result_df = pd.DataFrame(results, columns=['run_id', 'min_cluster_size', 'min_samples',
                                               'label_count', 'cost'])

    return result_df.sort_values(by='cost')


def plot_kmeans_inertia_and_silhouette(embeddings, k_range):
    '''
    Plot SSE and silhouette score for kmeans clustering for a range of k values
    Arguments:
        embeddings: array, sentence embeddings
        k_range: range, values of k to evaluate for kmeans clustering
    '''
    sse = []
    silhouette_avg_n_clusters = []

    for k in tqdm(k_range, total=len(k_range)):
        kmeans = sklearn.cluster.KMeans(n_clusters=k, init='k-means++', max_iter=100, n_init=1)
        kmeans.fit(embeddings)
        sse.append(kmeans.inertia_)

        silhouette_avg = sklearn.metrics.silhouette_score(embeddings, kmeans.predict(embeddings))
        silhouette_avg_n_clusters.append(silhouette_avg)

    # plot sse
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    axes[0].plot(k_range, sse)
    axes[0].set(xlabel='k clusters', ylabel='SSE', title='Elbow plot')
    axes[0].grid()

    # plot avg silhouette score
    axes[1].plot(k_range, silhouette_avg_n_clusters)
    axes[1].set(xlabel='k clusters', ylabel='Silhouette score', title='Silhouette score')
    axes[1].grid()

    plt.show()


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


def plot_confusion_metrics_and_print_class_mapping(labels, predicted_clusters, only_get_accuracy=False):
    classes_to_cluster_labels_random_mapping = dict(zip(set(labels), set(predicted_clusters)))
    confusion_metrics_labels_as_numbers = [classes_to_cluster_labels_random_mapping[class_original_label] for
                                           class_original_label in labels]
    output_confusion_matrix = confusion_matrix(confusion_metrics_labels_as_numbers,
                                               predicted_clusters)

    def _make_cost_m(output_confusion_matrix):
        s = np.max(output_confusion_matrix)
        return - output_confusion_matrix + s

    indexes = linear_assignment(_make_cost_m(output_confusion_matrix))
    old_to_new_predicted_cluster_mapping = dict(zip(indexes[0], indexes[1]))
    output_confusion_matrix[:, list(old_to_new_predicted_cluster_mapping.keys())] = output_confusion_matrix[:,
                                                                                    list(
                                                                                        old_to_new_predicted_cluster_mapping.values())]

    accuracy = np.trace(output_confusion_matrix) / np.sum(output_confusion_matrix)
    if only_get_accuracy:
        return accuracy
    ax = sns.heatmap(output_confusion_matrix, annot=True, fmt="d", cmap="Blues")
    ax.set_xlabel('Predicted labels')
    ax.set_ylabel('True labels')
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
        fig.add_trace(go.Bar(x=scores, y=words, orientation='h', marker_color=next(colors)), row=row, col=column)
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
                color="Black")},
        width=width * 4,
        height=height * rows if rows > 1 else height * 1.3,
        hoverlabel=dict(
            bgcolor="white",
            font_size=16,
            font_family="Rockwell"),
    )
    fig.update_xaxes(showgrid=True)
    fig.update_yaxes(showgrid=True)
    return fig


def plot_gt_clusters(embeddings, true_labels, n_neighbors=15, min_dist=0.1):
    umap_data = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          min_dist=min_dist,
                          # metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(len(embeddings))

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = true_labels

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color='lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.colorbar()
    plt.show()


def generate_clusters(embeddings,
                      umap_n_neighbors,
                      umap_n_components,
                      hdbscan_min_cluster_size,
                      random_state=None):
    """
    Generate HDBSCAN cluster object after reducing embedding dimensionality with UMAP
    """

    umap_embeddings = (umap.UMAP(n_neighbors=umap_n_neighbors,
                                 n_components=umap_n_components,
                                 metric='cosine',
                                 random_state=random_state)
                       .fit_transform(embeddings))

    clusters = hdbscan.HDBSCAN(min_cluster_size=hdbscan_min_cluster_size,
                               metric='euclidean',
                               cluster_selection_method='eom').fit(umap_embeddings)

    return clusters


def score_clusters(embeddings, clusters, prob_threshold=0.05):
    """
    Returns the label count and cost of a given clustering

    Arguments:
        clusters: HDBSCAN clustering object
        prob_threshold: float, probability threshold to use for deciding
                        what cluster labels are considered low confidence

    Returns:
        label_count: int, number of unique cluster labels, including noise
        cost: float, fraction of data points whose cluster assignment has
              a probability below cutoff threshold
    """
    cluster_labels = clusters.labels_
    label_count = len(np.unique(cluster_labels))
    total_num = len(clusters.labels_)
    cost = (np.count_nonzero(clusters.probabilities_ < prob_threshold) / total_num)
    silhouette = sklearn.metrics.silhouette_score(embeddings, cluster_labels)
    return label_count, cost, silhouette


def objective(params, embeddings, label_lower, label_upper):
    """
    Objective function for hyperopt to minimize

    Arguments:
        params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'random_state' and
               their values to use for evaluation
        embeddings: embeddings to use
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters

    Returns:
        loss: cost function result incorporating penalties for falling
              outside desired range for number of clusters
        label_count: int, number of unique cluster labels, including noise
        status: string, hypoeropt status

        """

    clusters = generate_clusters(embeddings,
                                 umap_n_neighbors=params['n_neighbors'],
                                 umap_n_components=params['n_components'],
                                 hdbscan_min_cluster_size=params['min_cluster_size'],
                                 random_state=params['random_state'])

    label_count, cost, silhouette = score_clusters(embeddings, clusters, prob_threshold=0.05)

    # 15% penalty on the cost function if outside the desired range of groups
    if (label_count < label_lower) | (label_count > label_upper):
        penalty = 0.15
    else:
        penalty = 0

    loss = cost + penalty

    return {'loss': loss, 'label_count': label_count, 'silhouette': silhouette, 'status': STATUS_OK}


def bayesian_search(embeddings, space, label_lower, label_upper, max_evals=100):
    """
    Perform bayesian search on hyperparameter space using hyperopt

    Arguments:
        embeddings: embeddings to use
        space: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', and 'random_state' and
               values that use built-in hyperopt functions to define
               search spaces for each
        label_lower: int, lower end of range of number of expected clusters
        label_upper: int, upper end of range of number of expected clusters
        max_evals: int, maximum number of parameter combinations to try

    Saves the following to instance variables:
        best_params: dict, contains keys for 'n_neighbors', 'n_components',
               'min_cluster_size', 'min_samples', and 'random_state' and
               values associated with lowest cost scenario tested
        best_clusters: HDBSCAN object associated with lowest cost scenario
                       tested
        trials: hyperopt trials object for search

        """

    trials = Trials()
    fmin_objective = partial(objective,
                             embeddings=embeddings,
                             label_lower=label_lower,
                             label_upper=label_upper)

    best = fmin(fmin_objective,
                space=space,
                algo=tpe.suggest,
                max_evals=max_evals,
                trials=trials)

    best_params = space_eval(space, best)
    print('best:')
    print(best_params)
    print(f"label count: {trials.best_trial['result']['label_count']}")

    best_clusters = generate_clusters(embeddings,
                                      umap_n_neighbors=best_params['n_neighbors'],
                                      umap_n_components=best_params['n_components'],
                                      hdbscan_min_cluster_size=best_params['min_cluster_size'],
                                      random_state=best_params['random_state'])

    return best_params, best_clusters, trials


def combine_results(df_ground, cluster_dict):
    """
    Returns dataframe of all documents and each model's assigned cluster

    Arguments:
        df_ground: dataframe of original documents with associated ground truth
                   labels
        cluster_dict: dict, keys as column name for specific model and value as
                      best clusters HDBSCAN object

    Returns:
        df_combined: dataframe of all documents with labels from
                     best clusters for each model

    """

    df_combined = df_ground.copy()

    for key, value in cluster_dict.items():
        df_combined[key] = value.labels_

    return df_combined


def summarize_results(results_dict, results_df):
    """
    Returns a table summarizing each model's performance compared to ground
    truth labels and the model's hyperparametes

    Arguments:
        results_dict: dict, key is the model name and value is a list of:
                      model column name in combine_results output, best_params and best_clusters
                      for each model (e.g. ['label_use', best_params_use, trials_use])
        results_df: dataframe output of combine_results function; dataframe of all documents
                    with labels from best clusters for each model

    Returns:
        df_final: dataframe with each row including a model name, calculated ARI and NMI,
                  loss, label count, and hyperparameters of best model

    """

    summary = []

    for key, value in results_dict.items():
        ground_label = results_df[RISK_1_COLUMN].values
        predicted_label = results_df[value[0]].values

        ari = np.round(adjusted_rand_score(ground_label, predicted_label), 3)
        nmi = np.round(normalized_mutual_info_score(ground_label, predicted_label), 3)
        loss = value[2].best_trial['result']['loss']
        label_count = value[2].best_trial['result']['label_count']
        n_neighbors = value[1]['n_neighbors']
        n_components = value[1]['n_components']
        min_cluster_size = value[1]['min_cluster_size']
        random_state = value[1]['random_state']

        summary.append([key, ari, nmi, loss, label_count, n_neighbors, n_components,
                        min_cluster_size, random_state])

    df_final = pd.DataFrame(summary, columns=['Model', 'ARI', 'NMI', 'loss',
                                              'label_count', 'n_neighbors',
                                              'n_components', 'min_cluster_size',
                                              'random_state'])

    return df_final.sort_values(by='NMI', ascending=False)


def plot_clusters(embeddings, clusters, n_neighbors=15, min_dist=0.1):
    """
    Reduce dimensionality of best clusters and plot in 2D

    Arguments:
        embeddings: embeddings to use
        clusteres: HDBSCAN object of clusters
        n_neighbors: float, UMAP hyperparameter n_neighbors
        min_dist: float, UMAP hyperparameter min_dist for effective
                  minimum distance between embedded points

    """
    umap_data = umap.UMAP(n_neighbors=n_neighbors,
                          n_components=2,
                          min_dist=min_dist,
                          # metric='cosine',
                          random_state=42).fit_transform(embeddings)

    point_size = 100.0 / np.sqrt(len(embeddings))

    result = pd.DataFrame(umap_data, columns=['x', 'y'])
    result['labels'] = clusters.labels_

    fig, ax = plt.subplots(figsize=(14, 8))
    outliers = result[result.labels == -1]
    clustered = result[result.labels != -1]
    plt.scatter(outliers.x, outliers.y, color='lightgrey', s=point_size)
    plt.scatter(clustered.x, clustered.y, c=clustered.labels, s=point_size, cmap='jet')
    plt.colorbar()
    plt.show()
