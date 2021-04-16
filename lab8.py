import pandas as pd
from numpy import unique
from numpy import where
from sklearn.cluster import KMeans
from sklearn.cluster import MiniBatchKMeans
from sklearn.cluster import MeanShift
from sklearn.cluster import OPTICS
from sklearn.cluster import SpectralClustering
from sklearn.mixture import GaussianMixture
from sklearn.cluster import AgglomerativeClustering
from sklearn.cluster import AffinityPropagation
from matplotlib import pyplot
from sklearn.decomposition import PCA
import seaborn as sns
import matplotlib.pyplot as plt

colors = ["red", "coral", "yellow", "green", "cyan", "blue", "violet", "gray", "purple"]


def k_means(data, X, n):
    model = KMeans(n_clusters=n)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def mini_batch_k_means(data, X, n):
    model = MiniBatchKMeans(n_clusters=n)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def mean_shift(data, X):
    model = MeanShift()
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def optics(data, X):
    model = OPTICS(eps=0.8, min_samples=10)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def spectral_clustering(data, X, n):
    model = SpectralClustering(n_clusters=n)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def gaussian_mixture_model(data, X, n):
    model = GaussianMixture(n_components=n)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def agglomerative_clustering(data, X, n):
    model = AgglomerativeClustering(n_clusters=n)
    yhat = model.fit_predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def affinity_propagation(data, X):
    model = AffinityPropagation(damping=0.9)
    model.fit(X)
    yhat = model.predict(X)
    clusters = unique(yhat)
    data['clusters'] = yhat
    print(data)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(X[row_ix, 0], X[row_ix, 1], alpha=0.4)
    pyplot.show()


def lab8(x_train):
    n = 2
    pca_num_components = 2

    # # анализ основных компонентов
    reduced_data = PCA(n_components=pca_num_components).fit_transform(x_train)

    k_means(x_train.copy(deep=True), reduced_data, n)
    mini_batch_k_means(x_train.copy(deep=True), reduced_data, n)
    mean_shift(x_train.copy(deep=True), reduced_data)
    optics(x_train.copy(deep=True), reduced_data)
    gaussian_mixture_model(x_train.copy(deep=True), reduced_data, n)
    agglomerative_clustering(x_train.copy(deep=True), reduced_data, n)
    affinity_propagation(x_train.copy(deep=True), reduced_data)


def lab8_tmin_all_data(x_train, y_train):
    pca_num_components = 2
    n = 6
    # # анализ основных компонентов
    reduced_data = PCA(n_components=pca_num_components).fit_transform(x_train)

    model = KMeans(n_clusters=n)
    x_train['clusters'] = model.fit_predict(x_train)
    print(x_train)
    yhat = x_train['clusters']
    clusters = unique(yhat)
    for cluster in clusters:
        row_ix = where(yhat == cluster)
        pyplot.scatter(reduced_data[row_ix, 0], reduced_data[row_ix, 1], alpha=0.4)
    pyplot.show()
