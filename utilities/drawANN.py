from graphviz import Digraph
from tqdm import tqdm

def draw_neural_network(layers):
    with tqdm(total=100) as pbar:
        
        g = Digraph('G', filename='neural_network.gv')
        g.attr(rankdir='TB', size='8,5')
        
        # create the input layer
        with g.subgraph(name='cluster_input') as c:
            c.attr(color='white')
            for i in range(layers[0].shape[0]):
                c.node('input{}'.format(i), 'input{}'.format(i))
        pbar.update(10)
        # create the hidden layers
        for i in range(len(layers)-1):
            with g.subgraph(name='cluster_hidden{}'.format(i)) as c:
                c.attr(color='white')
                for j in range(layers[i].shape[1]):
                    c.node('hidden{}{}'.format(i,j), 'hidden{}{}'.format(i,j))
        pbar.update(30)
        # create the output layer
        with g.subgraph(name='cluster_output') as c:
            c.attr(color='white')
            for i in range(layers[-1].shape[1]):
                c.node('output{}'.format(i), 'output{}'.format(i))
        pbar.update(10)
        # create the edges
        for i in range(layers[0].shape[0]):
            for j in range(layers[0].shape[1]):
                g.edge('input{}'.format(i), 'hidden0{}'.format(j))
        pbar.update(10)
        for i in range(len(layers)-1):
            for j in range(layers[i].shape[1]):
                for k in range(layers[i+1].shape[1]):
                    g.edge('hidden{}{}'.format(i,j), 'hidden{}{}'.format(i+1,k))
        pbar.update(30)
        for i in range(layers[-1].shape[1]):
            g.edge('hidden{}{}'.format(len(layers)-2,i), 'output{}'.format(i))
        pbar.update(10)
        
    return g