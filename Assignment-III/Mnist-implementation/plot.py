import matplotlib.pyplot as plt


def draw_plot(data, title, y_label, labels):

    x_values = [0.1, 0.5, 1.0, 10, 100, 1000]

    color_map = plt.get_cmap('Set1')
    num_colors = len(data[0])
    colors = [color_map(i/num_colors) for i in range(num_colors)]

    fig, ax = plt.subplots()

    for i in range(len(data)):
        ax.scatter([x_values[i]]*len(labels), data[i], marker='o', color=colors, label=i, s=150)

    ax.set_xscale('log') 
    ax.set_xticks(x_values)
    ax.set_xticklabels(x_values)

    ax.set_xlabel('Smoothing values')
    ax.set_ylabel(y_label)

    ax.set_title(title)

    handles, _ = ax.get_legend_handles_labels()

    handles = [plt.Line2D([], [], marker='o', color=colors[i], label=labels[i]) for i in range(len(labels))]
    ax.legend(handles=handles)

    plt.show()


accuracy_levels = [[0.7771, 0.7771, 0.8785], [0.6824, 0.6824, 0.867], [0.6428, 0.6428, 0.8647], [0.5921, 0.5921, 0.8618], [0.5859, 0.5859, 0.8619], [0.5851, 0.5851, 0.8619]]
accuracy_levels_for_logistic = [[0.8237, 0.8237, 0.8237], [0.8237, 0.8237, 0.8237], [0.8237, 0.8237, 0.8237], [0.8237, 0.8237, 0.8237], [0.8237, 0.8237, 0.8237], [0.8237, 0.8237, 0.8237]]
labels = ["Pixel Intensity", "Pca feature", "Hog feature"]
print(len(accuracy_levels))
draw_plot(accuracy_levels, "Accuracy analysis", "Accuracy", labels)