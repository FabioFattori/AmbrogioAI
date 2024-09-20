from graphviz import Digraph

def draw_neural_network(layers):
    
    
    
    g = Digraph(format='png')
    g.attr(rankdir='LR', size='12',dpi='600')
    with g.subgraph(name='cluster_'+str(0)) as c:
            c.attr(color='blue')
            c.attr(label='Input Layer')
            # draw a circle with the first and the last neuron of the input layer
            c.node('{}'.format(layers[0][0].id), shape='circle')
            c.node('{}'.format(layers[0][len(layers[0])-1].id), shape='circle')
            # put this line to avoid the first and the last neuron to be connected
            g.edge('{}'.format(layers[0][0].id), '{}'.format(layers[0][len(layers[0])-1].id), style='invis')
    
    inputLayer = layers.pop(0)
    
    # Add nodes for each layer
    for i, layer_size in enumerate(layers):
        with g.subgraph(name='cluster_'+str(i+1)) as c:
            c.attr(color='blue')
            
            if(i == -1):
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
    
    
    for j in range(len(layers[0])):
        g.edge('{}'.format(inputLayer[0].id), '{}'.format(layers[0][j].id))
        g.edge('{}'.format(inputLayer[4095].id), '{}'.format(layers[0][j].id))
        
    return g