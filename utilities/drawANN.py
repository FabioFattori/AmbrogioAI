from graphviz import Digraph

def draw_neural_network(layers):
    g = Digraph(format='png')
    g.attr(rankdir='LR', size='10,8',dpi='600')
    
    # Add nodes for each layer
    for i, layer_size in enumerate(layers):
        with g.subgraph(name='cluster_'+str(i)) as c:
            c.attr(color='blue')
            
            if(i == 0):
                c.attr(label='Input Layer')
            elif(i == len(layers) - 1):
                c.attr(label='Output Layer')
            else:
                c.attr(label='Hidden Layer {}'.format(i))
            
            for j in range(len(layer_size)):
                c.node('{}'.format(layer_size[j].id), shape='circle')
    
    # Add edges between layers
    for i in range(len(layers) - 1):
        for j in range(len(layers[i])):
            for k in range(len(layers[i+1])):
                g.edge('{}'.format(layers[i][j].id), '{}'.format(layers[i+1][k].id))
    
    return g