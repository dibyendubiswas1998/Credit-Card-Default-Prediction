import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from kneed import KneeLocator
from src.utils.common_utils import log



def elbow_method(data, log_file, plot_path):
    """
        It helps to find the number of cluster.\n
        :param data: data
        :param log_file: log_file
        :param plot_path: plot_path
        :return: no_of_cluster
    """
    try:
        file = log_file
        wcss = []
        for i in range(1, 20):
            kmeans = KMeans(n_clusters=i, init='k-means++', random_state=42)  # initializing the KMeans object
            kmeans.fit(data)  # fitting the data to the KMeans Algorithm
            wcss.append(kmeans.inertia_)
        plt.plot(range(1, 11), wcss)  # creating the graph between WCSS and the number of clusters
        plt.title('The Elbow Method')
        plt.xlabel('Number of clusters')
        plt.ylabel('WCSS')
        plt.savefig(plot_path)
        log(file_object=file, log_message=f"save the elbow plot in {plot_path}") # logs the details

        # finding the value of the optimum cluster programmatically
        kn = KneeLocator(range(1, 20), wcss, curve='convex', direction='decreasing')
        log(file_object=file, log_message=f"get the number of cluster {kn}") # logs the details
        return kn # return the number of cluster.

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


def create_clustering(data, number_of_clusters, log_file, random_state):
    """
        It helps to create cluster.\n
        :param data:  data
        :param number_of_clusters: number_of_clusters
        :param log_file: log_file
        :param random_state: random_state
        :return: clustered data.
    """
    try:
        file = log_file
        kmeans_model = KMeans(n_clusters=number_of_clusters, init='k-means++', random_state=random_state)
        y_kmeans = kmeans_model.fit_predict(data)  # divide data into clusters
        log(file_object=file, log_message=f"create the {number_of_clusters} cluster based on KMeans++ & random_state: {random_state}")

        data['cluster'] = y_kmeans # labeled the cluster data
        return data, kmeans_model # return clustered data & cluster model.

    except Exception as e:
        print(e)
        file = log_file
        log(file_object=file, log_message=f"Error will be: {e}")  # logs the details
        raise e


if __name__ == '__main__':
    pass
