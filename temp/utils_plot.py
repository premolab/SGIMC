import numpy as np
import matplotlib.pyplot as plt

def plot_WH(W, H):
    fig = plt.figure(figsize=(5, 10))
    ax = fig.add_subplot(121, title="$W$", xticks=[])
    ax.imshow(np.concatenate(
        [W, np.full_like(W[:, :1], W.min()),
         np.where(np.linalg.norm(W, axis=1, keepdims=True) > 0, W.max(), W.min())],
        axis=-1), cmap=plt.cm.hot)

    ax = fig.add_subplot(122, title="$H$", xticks=[])
    ax.imshow(np.concatenate(
        [H, np.full_like(H[:, :1], H.min()),
         np.where(np.linalg.norm(H, axis=1, keepdims=True) > 0, H.max(), H.min())],
        axis=-1), cmap=plt.cm.hot)

    fig.tight_layout()
    plt.show()
    plt.close()
    

def plot_loss(losses, titles, y_names, fig_size=4,
              max_cols=None):
    
    assert len(losses) == len(titles) == len(y_names)
    
    if max_cols is None:
        rows, columns = 1, len(losses)
    else:
        columns = min(len(losses), max_cols)
        rows = (len(losses) - 1) // columns + 1

    n_iters = np.array(
        [ np.array(range(len(losses[i]))) + 1 for i in range(len(losses)) ]
    )

    fig, axes = plt.subplots(rows, columns,
                             figsize=(fig_size * columns, fig_size * rows))
    axes = np.array(axes).flatten()

    for ax, n_iter, loss, title, norm in zip(axes, n_iters, losses, titles, y_names):
        ax.plot(n_iter, loss, 'r-', linewidth=2)
        ax.set_ylabel(norm)
        ax.set_xlabel('Number of iteration')
        ax.set_title(title)


    plt.tight_layout()
    plt.show()
    plt.close()