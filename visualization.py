import networkx as nx
import seaborn as sns
import matplotlib.pyplot as plt

'''
    CREATING PLOT OF THE GRAPH

    input: graph and colors
    1. get the position attribute for each node
    2. assign the colors to the nodes based on their cluster labels
    3. With the conditional statement checks if colors (c) have been assigned to the nodes. If colors have been assigned, it draws the graph with node colors specified by c. 
        Otherwise, it draws the graph without specifying node colors
    4. Displays the plot without blocking the program
'''

def plot2d_graph(graph, colors):
    pos = nx.get_node_attributes(graph, 'pos')
    c = [colors[i % (len(colors))]
         for i in nx.get_node_attributes(graph, 'cluster').values()]
    if c:  # is set
        nx.draw(graph, pos, node_color=c, node_size=0.25)
    else:
        nx.draw(graph, pos, node_size=0.25)
    plt.show(block=False)


'''
    CREATING SCATTER PLOT OF DATA in 2D

    input: dataframe
    1. if number of columns is greater than 3 there's a warning (maybe I am using the wrong columns for the scatter plot)
    2. building a scatter plot, the color of each point is assigned according to its cluster id
        I use the first two columns for the x and y coordinates
    3. displays the plot (no blocking window)
    '''

def plot2d_data(df):
    if (len(df.columns) > 3):
        print("Plot Waring: more than 2-Dimensions!")
    df.plot(kind='scatter', c=df['cluster'], cmap='gist_rainbow', x=0, y=1)
    plt.show(block=False)