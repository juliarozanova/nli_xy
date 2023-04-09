from prefect import task
from sklearn.decomposition import PCA

@task
def pca(reps):
    pca = PCA(n_components=3)
    pca.fit(reps)
    reps_reduced = pca.transform(reps)

    return {
        'reps_reduced': reps_reduced,
        'pca': pca
    }
