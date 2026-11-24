import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

def plot_gantt_charts(comparison):

    if not comparison:
        raise ValueError("No schedular data provided!!!")

    all_segments = [segment for metrics in comparison.values() for segment in metrics["gantt"]]

    if not all_segments:
        raise ValueError("No gantt chart segments found :(")

    max_time = max(segment["end"] for segment in all_segments)

    # each processes gets different color throughout
    process_ids = list(dict.fromkeys(segment["pid"] for segment in all_segments if segment["pid"] != "IDLE"))

    cmap = plt.get_cmap("tab20")

    process_colors = {pid: cmap(i%cmap.N) for i, pid in enumerate(process_ids)}

    algorithms = list(comparison.keys())

    fig, axes = plt.subplots(len(algorithms), 1, figsize = (14, 2* len(algorithms)), sharex = True)

    if len(algorithms) == 1:
        axes = [axes]

    for ax, algorithm in zip(axes, algorithms):
        gantt = comparison[algorithm]["gantt"]

        for segment in gantt:
            pid = segment["pid"]
            start = segment["start"]
            end = segment["end"]

            duration = end - start

            color = ("D3D3D3" if pid == "IDLE" else process_colors[pid])

            ax.barh(y=0,
                    width = duration,
                    left = start,
                    height = 0.6,
                    color = color,
                    edgecolor = "black",
                    linewidth = 0.8)

            # displaying process id inside the bar
            if duration >= 0.75:
                ax.text((start+end)/2,
                        0,
                        str(pid),
                        ha = "center",
                        va = "center",
                        fontsize = 8)

        ax.set_ylabel(algorithm,
                      rotation = 0,
                      labelpad = 70,
                      va = "center")

        ax.set_yticks([])
        ax.set_ylim(-0.6, 0.6)
        ax.set_xlim(0, max_time)

        ax.grid(axis = "x",
                linestyle = "--",
                alpha = 0.35)

        ax.set_axisbelow(True)

        ax.spines["top"].set_visible(False)
        ax.spines["right"].set_visible(False)
        ax.spines["left"].set_visible(False)

    axes[-1].set_xlabel("time units")
    axes[-1].xaxis.set_major_locator(MaxNLocator(integer = True))

    fig.suptitle("CPU scheduling algorithm comparison", fontsize = 14)

    fig.tight_layout
    plt.show()