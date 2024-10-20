"""A class to generate plots for the results of applied functions, loss, AUC, Accuracy, Precesion etc,
 of models trained with machine learning methods.
Example:
    plotter = Plotter(functions= ["Accuracy", "loss", "AUC"])
    for epoch in range(100):
        loss_train, acc_train, auc_train = your_model.train()
        loss_val, acc_val, auc_val = your_model.validate()
        plotter.add_values(epoch, [acc_train, loss_train, auc_train],[val_acc, loss_val, val_auc])

    plotter.block()
# if you dont have any of these values use None instead
Example, only train loss chart:
    plotter = Plotter(functions= ["Loss"])
    for epoch in range(100):
        loss_train = your_model.train()
        plotter.add_values(epoch, [loss_train], [None])
    plotter.block()

"""
from __future__ import absolute_import
from __future__ import print_function
import os
import matplotlib.pyplot as plt
import numpy as np
import warnings
import math
import networkx as nx
from matplotlib.ticker import MaxNLocator
def ignore_nan_and_inf(value, label, x_index):
    """Helper function that creates warnings on NaN/INF and converts them to None.
    Args:
        value: The value to check for NaN/INF.
        label: For which line the value was used (usually "loss train", "loss val", ...)
            This is used in the warning message.
        x_index: At which x-index the value was used (e.g. 1 as in Epoch 1).
            This is used in the warning message.
    Returns:
        value, but None if value is NaN or INF.
    """
    if value is None:
        return None
    elif math.isnan(value):
        warnings.warn("Got NaN for value '%s' at x-index %d" % (label, x_index))
        return None
    elif math.isinf(value):
        warnings.warn("Got INF for value '%s' at x-index %d" % (label, x_index))
        return None
    else:
        return value


class Plotter(object):
    """Class to plot loss and accuracy charts (for training and validation data)."""

    def __init__(self,
                 title=None,
                 save_to_filepath=None,
                 functions= ["Accuracy", "-Elbo", "AUC"#,"AP"
                 ],
                 show_plot_window=True,
                 x_label="Epoch"):
        """Constructs the plotter.
        Args:
            title: An optional title which will be shown at the top of the
                plot. E.g. the name of the experiment or some info about it.
                If set to None, no title will be shown. (Default is None.)
            functions: name of the all functions applied to train and test set and
                will be plotted.
            save_to_filepath: The path to a file in which the plot will be saved,
                e.g. "/tmp/last_plot.png". If set to None, the chart will not be
                saved to a file. (Default is None.)
            show_plot_window: Whether to show the plot in a window (True)
                or hide it (False). Hiding it makes only sense if you
                set save_to_filepath. (Default is True.)
            x_label: Label on the x-axes of the charts. Reasonable choices
                would be: "Epoch", "Batch" or "Example". (Default is "Epoch".)
        """

        assert save_to_filepath is not None or show_plot_window

        self.title = title
        self.title_fontsize = 14
        self.show_plot_window = show_plot_window
        self.save_to_filepath = save_to_filepath
        self.x_label = x_label
        self.functions = functions
        self.x_value = []
        # alpha values
        # 0.8 = quite visible line
        # 0.5 = moderately visible line
        # thick is used for averages and regression (also for the main values,
        # if there are no averages),
        # thin is used for the main values
        self.alpha_thick = 0.8
        self.alpha_thin = 0.5




        # whether to show grids in both charts
        self.grid = True

        # the styling of the lines
        # sma = simple moving average
        self.linestyles = {
            "train": "r-",
            "val": "b-",
        }
        # different linestyles for the first epoch (if only one value is available),
        # because no line can then be drawn (needs 2+ points) and only symbols will
        # be shown.
        self.linestyles_one_value = {
            "train": "rs-",
            "val": "b^-"
        }

        # these values will be set in _initialize_plot() upon the first call
        # of redraw()
        # fig: the figure of the whole plot
        # ax_loss: loss chart (left)
        # ax_acc: accuracy chart (right)
        self.fig = None
        self.ax_loss = None
        self.ax_acc = None

        # dictionaries with x, y values for each line
        self.values_train = []
        self.values_validation = []
        for x in functions:
            self.values_train.append([])
            self.values_validation.append([])

    def add_values(self, x_index, train_values=[], validation_values=[], redraw=True):
        """Function to add new values for each line for a specific x-value (e.g.
        a specific epoch).
        Meaning of the values / lines:
         - train_values: y-value of the functions applied to the training set.
         - train_values:   y-value of the functions applied to the validation set.

        Values that are None will be ignored.
        Values that are INF or NaN will be ignored, but create a warning.
        It is currently assumed that added values follow logically after
        each other (progressive order), so the first x_index might be 1 (first entry),
        then 2 (second entry), then 3 (third entry), ...
        Not allowed would be e.g.: 10, 11, 5, 7, ...
        If that is not the case, you will get a broken line graph.
        Args:
            x_index: The x-coordinate, e.g. x_index=5 might represent Epoch 5.
            loss_train: a list contain the y-value of the functions at the given x_index.
                If None, no value for the loss train line will be added at
                the given x_index. (Default is None.) The valuse should have the same
                order with the functions
            loss_val: Same as loss_train for the validation line.
                (Default is None.)
            redraw: Whether to redraw the plot immediately after receiving the
                new values. This is reasonable if you add values once at the end
                of every epoch. If you add many values in a row, set this to
                False and call redraw() at the end (significantly faster).
                (Default is True.)
        """
        assert isinstance(x_index, (int))

        # loss_train = ignore_nan_and_inf(loss_train, "loss train", x_index)
        # loss_val = ignore_nan_and_inf(loss_val, "loss val", x_index)
        # acc_train = ignore_nan_and_inf(acc_train, "acc train", x_index)
        # acc_val = ignore_nan_and_inf(acc_val, "acc val", x_index)
        self.x_value.append(x_index)
        for i, x in enumerate(validation_values):
            self.values_validation[i].append(x)
        for i, x in enumerate(train_values):
            self.values_train[i].append(x)

        if redraw:
            self.redraw()

    def block(self):
        """Function to show the plot in a blocking way.
        This should be called at the end of your program. Otherwise the
        chart will be closed automatically (at the end).
        By default, the plot is shown in a non-blocking way, so that the
        program continues execution, which causes it to close automatically
        when the program finishes.
        This function will silently do nothing if show_plot_window was set
        to False in the constructor.
        """
        if self.show_plot_window:
            plt.figure(self.fig.number)
            plt.show()

    def save_plot(self, filepath):
        """Saves the current plot to a file.
        Args:
            filepath: The path to the file, e.g. "/tmp/last_plot.png".
        """
        self.fig.savefig(filepath, bbox_inches="tight")

    def _initialize_plot(self):
        """Creates empty figure and axes of the plot and shows it in a new window.
        """
        fig, self.axes = plt.subplots(nrows=len(self.functions)//2+1 , ncols=2 , figsize=(30, 8))
        self.fig = fig

        if len(self.functions) == 1:
            self.axes = np.expand_dims(self.axes, axis=0)
            # self.axes = [self.axes]

        if len(self.functions) % 2 != 0 :
            self.fig.delaxes(self.axes[len(self.functions)//2, 1])


        # set_position is neccessary here in order to make space at the bottom
        # for the legend
        for row in self.axes:
            for col in row:
                if col is not None:
                    box = col.get_position()
                    col.set_position([box.x0, box.y0 + box.height * 0.1,
                                 box.width, box.height * 0.77])

        # draw the title
        # it seems to be necessary to set the title here instead of in redraw(),
        # otherwise the title is apparently added again and again with every
        # epoch, making it ugly and bold
        if self.title is not None:
            self.fig.suptitle(self.title, fontsize=self.title_fontsize)

        if self.show_plot_window:
            plt.show(block=False)

    def redraw(self):
        """Redraws the plot with the current values.
        It should not be called many times per second as that would be slow.
        Calling it every couple seconds should create no noticeable slowdown though.
        """
        # initialize the plot if it's the first redraw
        if self.fig is None:
            self._initialize_plot()

        # activate the plot, in case another plot was opened since the last call
        plt.figure(self.fig.number)



        # set chart titles, x-/y-labels and grid
        j = 0
        for row in self.axes:
            for col in row:
                if col and j<len(self.functions):
                    col.clear()
                    col.set_title(self.functions[j])
                    col.set_ylabel(self.functions[j])
                    col.set_xlabel(self.x_label)
                    col.grid(self.grid)
                    j = j+1
        # Plot main lines, their averages and the regressions (predictions)
        self._redraw_main_lines()


        # Add legends (below both chart)
        ncol = 1
        labels = ["$CHART train", "$CHART val."]
        

        j = 0
        for row in self.axes:
            for col in row:
               if j<len(self.functions):
                        col.legend([lab.replace("$CHART", self.functions[j]) for lab in labels],
                                   loc="upper center",
                                   bbox_to_anchor=(0.5, -0.2),
                                   ncol=len(self.functions))
                        j=j+1
        import matplotlib
        # matplotlib.pyplot.ion()
        # plt.show(block=True)
        # matplotlib.get_backend()
        # matplotlib.use('TkAgg')
        # plt.show(block=True)
        # plt.interactive(True)
        plt.draw()
        # plt.draw()
        # plt.show()
        # plt.show(block=True)
        # plt.show()
        plt.pause(0.1)
        # save the redrawn plot to a file upon every redraw.
        if self.save_to_filepath is not None:
            self.save_plot(self.save_to_filepath)


    def _redraw_main_lines(self):
        """Draw the main lines of values (i.e. loss train, loss val, acc train, acc val).
        Returns:
            List of handles (one per line).
        """
        handles = []
        i = 0
        for row in self.axes:
            for col in row:
                # Set the styles of the lines used in the charts
                # Different line style for epochs after the first one, because
                # the very first epoch has only one data point and therefore no line
                # and would be invisible without the changed style.

                train_style = self.linestyles["train"]
                val_style = self.linestyles["val"]

                if len(self.values_train[0]) == 1:
                    train_style = self.linestyles_one_value["train"]
                    val_style = self.linestyles_one_value["val"]


                alpha_main = self.alpha_thin


                # Plot the lines
                if col and i < len(self.functions):
                    #removing None element
                    index = np.array(self.values_train[i])!=None

                    h_lt, = col.plot(np.array(self.x_value)[index], np.array(self.values_train[i])[index],
                                 train_style, label=self.functions[i]+" train", alpha=alpha_main)
                    index = np.array(self.values_validation[i])!=None
                    h_lv, = col.plot( np.array(self.x_value)[index], np.array(self.values_validation[i])[index],
                                 val_style, label=self.functions[i]+" val.", alpha=alpha_main)
                    handles.extend([h_lt, h_lv])
                    if self.functions[i]== "Accuracy":
                        col.xaxis.set_major_locator(MaxNLocator(integer=True))
                i = i+1
        return handles




# class Hist_Plotter():
#     def __init__(self):
#         plt.close(def__fig_num)
def hist_plotter(p, q, p_labal = "org_data dist", q_label="generated_dist" ):
        plt.close("HIST")
        plt.figure("HIST")
        max_degree = np.argwhere(p>0)[-1][0]
        max_degree = max_degree if max_degree > np.argwhere(q>0)[-1][0] else np.argwhere(q>0)[-1][0]


        p = p[: max_degree]
        q= q[: max_degree]
        plt.bar(height=p, x=list(range(len(p))), color=(0.3,0.1,0.4,0.6), label=p_labal)
        plt.bar(height=q, x=list(range(len(q))), color=(0.3,0.5,0.4,0.6), label= q_label)
        plt.grid(axis='y', alpha=0.75)
        plt.xlabel('Value')
        plt.ylabel('Frequency')
        plt.legend(loc='upper right')


        plt.draw()
        plt.pause(0.05)

def plotG(G, type, graph_name = "Generated Graph", file_name=None, plot_it=False):
    plt.close(graph_name)

    pos = nx.spring_layout(G, iterations=1000)
    f = plt.figure(graph_name)
    nx.draw(G, pos, node_size=20, width=1, edge_color="black")

    # plt.draw()

    # plt.show()
    # plt.pause(0.5)
    f.savefig(type+ "_graph.png" if file_name == None else file_name)
    if plot_it: plt.show()
    #
def plot_both( origianl_graphs,generated_graphs, origi_i, gen_i, plot_name="original_generated"):
    """
        origianl_graphs: a list of graphs ehich are fiven as training
        generated_graphs: a list of graphs which are generated
        gen_i: prediction for new description, a part of graph which has not seen and recoved, it will be in red
        origi_i: the thing which should be predicted
    """



    import matplotlib.pyplot as plt
    import networkx as nx
    #ToDo:clean it up
    plt.close(plot_name)
    plt.figure(plot_name)

    counter =0
    for i in range(len( origianl_graphs)):
        generated =generated_graphs[i]
        orig = origianl_graphs[i]
        counter+=1
        generated[generated>1] = 1
        orig[orig>1]=1
        generated = nx.from_numpy_array(generated)
        orig = nx.from_numpy_array(orig)
        pos = nx.spring_layout(generated, iterations=1000)
        plt.subplot(220+counter)
        nx.draw(generated, pos, node_size=50, width=0.1)
        if i!=len( origianl_graphs)-1:
            G = nx.from_numpy_array(gen_i[i])
            nx.draw_networkx_edges(generated,pos,edgelist=G.edges(),edge_color="r",)
        counter += 1
        pos = nx.spring_layout(orig, iterations=1000)
        plt.subplot(220+counter)
        nx.draw(orig, pos, node_size=50, width=0.1)
        if i != len(origianl_graphs) - 1:
            G = nx.from_numpy_array(origi_i[i])
            nx.draw_networkx_edges(generated, pos, edgelist=G.edges(), edge_color="r", )


    plt.show()

# def featureVisualizer(features, node_color,node_label, lr = 10,per = 100, n_iter=5000,n_components=2,metric="cosine", fig_name=""):
#     cm = plt.cm.get_cmap('RdYlBu')
#     # Apply t-SNE transformation on node embeddings
#     from sklearn.manifold import TSNE
#     tsne = TSNE(learning_rate=lr, metric=metric,perplexity = per, n_components = n_components, verbose = 1, n_iter = n_iter, init ="pca")
#     node_embeddings_2d = tsne.fit_transform(features)
#     f = plt.figure(fig_name+"RepresentaionPlot", figsize=(10, 8))
#     import matplotlib as matplotlib
#     import numpy as np
#     import matplotlib.pyplot as plt
#
#     def cmap_map(function, cmap):
#         """ Applies function (which should operate on vectors of shape 3: [r, g, b]), on colormap cmap.
#         This routine will break any discontinuous points in a colormap.
#         """
#         cdict = cmap._segmentdata
#         step_dict = {}
#         # Firt get the list of points where the segments start or end
#         for key in ('red', 'green', 'blue'):
#             step_dict[key] = list(map(lambda x: x[0], cdict[key]))
#         step_list = sum(step_dict.values(), [])
#         step_list = np.array(list(set(step_list)))
#         # Then compute the LUT, and apply the function to the LUT
#         reduced_cmap = lambda step: np.array(cmap(step)[0:3])
#         old_LUT = np.array(list(map(reduced_cmap, step_list)))
#         new_LUT = np.array(list(map(function, old_LUT)))
#         # Now try to make a minimal segment definition of the new LUT
#         cdict = {}
#         for i, key in enumerate(['red', 'green', 'blue']):
#             this_cdict = {}
#             for j, step in enumerate(step_list):
#                 if step in step_dict[key]:
#                     this_cdict[step] = new_LUT[j, i]
#                 elif new_LUT[j, i] != old_LUT[j, i]:
#                     this_cdict[step] = new_LUT[j, i]
#             colorvector = list(map(lambda x: x + (x[1],), this_cdict.items()))
#             colorvector.sort()
#             cdict[key] = colorvector
#
#         return matplotlib.colors.LinearSegmentedColormap('colormap', cdict, 1024)
#
#     light_jet = cmap_map(lambda x: x / 2 + 0.5, matplotlib.cm.jet)
#     plt.scatter(
#         node_embeddings_2d[:, 0],
#         node_embeddings_2d[:, 1],
#         c=np.array(node_color) / 30 + 1, cmap=light_jet,
#         alpha=0.7,
#     )
#     # plt.scatter(
#     #     node_embeddings_2d[:, 0],
#     #     node_embeddings_2d[:, 1],
#     #     c=node_color, vmin=0, vmax=400,
#     #     cmap=cm,
#     # alpha = 0.7,
#     # )
#     # plt.show()
#     if node_label!=None:
#         for i, txt in enumerate(node_label):
#             plt.annotate(txt, (node_embeddings_2d[:, 0][i], node_embeddings_2d[:, 1][i]))
#     f.savefig(fig_name+"_tsne_graph.png" )


import matplotlib.pyplot as plt
def featureVisualizer(features, node_color, filename="", lr = 10,per = 100, n_iter=5000,n_components=2,metric="cosine", legend_label=None):
    # Apply t-SNE transformation on node embeddings
    from sklearn.manifold import TSNE
    tsne = TSNE(learning_rate=lr, metric=metric,perplexity = per, n_components = n_components, verbose = 1, n_iter = n_iter, init ="pca")
    node_embeddings_2d = tsne.fit_transform(features)

    f = plt.figure(filename+"_RepresentaionPlot", figsize=(10, 8))

    # plt.legend(handles=scatter.legend_elements()[0], labels=legend_label, fontsize="x-large", loc='upper right')
    if legend_label!=None:
        scatter = plt.scatter(
            node_embeddings_2d[:, 0],
            node_embeddings_2d[:, 1],
            c=node_color,
            cmap="jet",
            alpha=0.7,
        )
        plt.legend(handles=scatter.legend_elements()[0], labels=legend_label, fontsize=22, loc='upper right')
    else:
        plt.scatter(
            node_embeddings_2d[:, 0],
            node_embeddings_2d[:, 1],
            c=node_color,
            cmap="jet",
            alpha=0.7,
        )
        plt.show()

    # plt.show()
    # if node_label!=None:
    #     for i, txt in enumerate(node_label):
    #         plt.annotate(txt, (node_embeddings_2d[:, 0][i], node_embeddings_2d[:, 1][i]),fontsize=6)

    f.savefig(filename+"_tsne_graph.png" )


if __name__ == '__main__':
    # import scipy
    # import pickle
    #
    # # load data; include node embedding and edges label
    # dataset = "ACM"
    # with open(dataset + '_edge_label.npz', 'rb') as infile:
    #     adj_matrix = pickle.load(infile)
    #
    # features = np.load(dataset + '_node_features.npy')
    #
    # # ploting the edges based on node representation
    # edgefeatures = np.load(dataset + '_edge_features.npy')
    # edges_feature = []
    # for i,j in zip(*adj_matrix.nonzero()):
    #     feature = np.array([f_i[i,j] for f_i in edgefeatures])
    #     edges_feature.append(feature)
    #
    #
    #
    # pair_features = []
    # label = []
    # for i,j in zip(*adj_matrix.nonzero()):
    #     pair_features.append(features[i]+features[j])
    #     label.append(adj_matrix[i,j])
    #
    # # plot edge embedding for dyed
    # featureVisualizer(pair_features, label,label)
    # featureVisualizer(edges_feature, label,label)




    G = nx.grid_2d_graph(10, 10)
    plotG(G, "")
