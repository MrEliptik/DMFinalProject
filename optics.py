import numpy as np
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
from sklearn.cluster import OPTICS

def cluster(data, min_samples=5, eps=0.5 ,reachability_plot=True, clustering_visualization=False):
        clust = OPTICS(min_samples=min_samples, rejection_ratio=eps)

        # Run the fit
        clust.fit(data)

        space = np.arange(len(data))
        reachability = clust.reachability_[clust.ordering_]
        labels = clust.labels_[clust.ordering_]

        plt.figure(figsize=(10, 7))
        G = gridspec.GridSpec(2, 3)
        ax1 = plt.subplot(G[0, :])
        if(clustering_visualization):
                ax2 = plt.subplot(G[1, 0])

        if(reachability_plot):
                # Reachability plot
                color = ['g.', 'r.', 'b.', 'y.', 'c.']
                for k, c in zip(range(0, 5), color):
                        Xk = space[labels == k]
                        Rk = reachability[labels == k]
                        ax1.plot(Xk, Rk, c, alpha=0.3)
                ax1.plot(space[labels == -1], reachability[labels == -1], 'k.', alpha=0.3)
                ax1.plot(space, np.full_like(space, 0.75, dtype=float), 'k-', alpha=0.5)
                ax1.plot(space, np.full_like(space, 0.25, dtype=float), 'k-.', alpha=0.5)
                ax1.set_ylabel('Reachability (epsilon distance)')
                ax1.set_title('Reachability Plot')
        
        if(clustering_visualization):
                x = np.array(data)
                # OPTICS
                color = ['g.', 'r.', 'b.', 'y.', 'c.']
                for k, c in zip(range(0, 5), color):
                        Xk = x[clust.labels_ == k]
                        ax2.plot(Xk[:, 0], Xk[:, 1], c, alpha=0.3)
                ax2.plot(x[clust.labels_ == -1, 0], x[clust.labels_ == -1, 1], 'k+', alpha=0.1)
                ax2.set_title('Automatic Clustering\nOPTICS')
        
        plt.tight_layout()
        plt.show()

        return clust