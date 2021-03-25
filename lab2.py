import seaborn as sns
import matplotlib.pyplot as plt


def lab2(data):
    sns.violinplot(x=data.positive_ratings_percentage)
    sns.displot(data.positive_ratings_percentage)
    plt.show()
    sns.boxplot(x=data.positive_ratings_percentage)
    plt.show()

    sns.pairplot(data[['english', 'required_age', 'achievements', 'positive_ratings_percentage', 'owners',
                       'median_playtime', 'price']])
    plt.show()
    print(data[['english', 'required_age', 'achievements', 'positive_ratings', 'negative_ratings', 'average_playtime',
                'median_playtime', 'owners', 'price', 'positive_ratings_percentage']].corr())
    print(data.describe())
