import matplotlib.pyplot as plt

# CAUSE (bug): the old code imported IPython.display unconditionally and used
# it to redraw the plot, which crashes with an ImportError in headless/SSH
# sessions (no IPython / no display backend).
# WHY: matplotlib alone can render interactively (plt.pause) when a GUI
# backend is available, and can always save the chart to a file as a last
# resort — so training never dies just because of plotting.
try:
    from IPython import display  # noqa: F401  (optional, only used if present)
    HAS_IPYTHON = True
except ImportError:
    HAS_IPYTHON = False

plt.ion()


def plot(scores, mean_scores):
    plt.clf()
    plt.title("Training...")
    plt.xlabel("Number of Games")
    plt.ylabel("Score")
    plt.plot(scores)
    plt.plot(mean_scores)
    plt.ylim(ymin=0)
    plt.text(len(scores) - 1, scores[-1], str(scores[-1]))
    plt.text(len(mean_scores) - 1, mean_scores[-1], str(mean_scores[-1]))

    if HAS_IPYTHON:
        # original behaviour: live redraw inside Jupyter/IPython
        display.clear_output(wait=True)
        display.display(plt.gcf())
    else:
        # CAUSE (bug): plt.ion() alone doesn't redraw when there is no event
        # loop running.
        # WHY: plt.pause() pumps the GUI event loop so the window updates; if
        # even that fails (pure headless), we save the figure to disk instead
        # of crashing the training loop.
        try:
            plt.pause(0.001)
        except Exception:
            plt.savefig("training_progress.png")
