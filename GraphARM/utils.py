import torch
from torch_geometric.utils import to_dense_adj
from torch_geometric.data import Data

def random_node_decay_ordering(datapoint):
    # create random list of nodes
    return torch.randperm(datapoint.x.shape[0]).tolist()

class NodeMasking:
    def __init__(self, dataset):
        self.dataset = dataset
        
        # Handle PyTorch Geometric Dataset objects
        if hasattr(dataset, 'data') and hasattr(dataset, 'x'):
            # This is a PyTorch Geometric Dataset object
            assert dataset.x.shape[1] == 1, "Only one feature per node is supported"
            
            self.NODE_MASK = dataset.x.unique().shape[0]
            self.EMPTY_EDGE = dataset.edge_attr.unique().shape[0]
            self.EDGE_MASK = dataset.edge_attr.unique().shape[0] + 1
        elif isinstance(dataset, list):
            # For list of Data objects, analyze the first one to get statistics
            sample_data = dataset[0]
            assert sample_data.x.shape[1] == 1, "Only one feature per node is supported"
            
            # Get unique values across all data in the dataset
            all_x = torch.cat([data.x for data in dataset])
            all_edge_attr = torch.cat([data.edge_attr for data in dataset])
            
            self.NODE_MASK = all_x.unique().shape[0]
            self.EMPTY_EDGE = all_edge_attr.unique().shape[0]
            self.EDGE_MASK = all_edge_attr.unique().shape[0] + 1
        else:
            # For single Data object
            assert dataset.x.shape[1] == 1, "Only one feature per node is supported"
            
            self.NODE_MASK = dataset.x.unique().shape[0]
            self.EMPTY_EDGE = dataset.edge_attr.unique().shape[0]
            self.EDGE_MASK = dataset.edge_attr.unique().shape[0] + 1
    
    def idxify(self, datapoint):
        '''
        Converts node and edge types to indices starting from 0
        '''
        datapoint = datapoint.clone()
        unique_node_types = {node_type.item(): idx for idx, node_type in enumerate(datapoint.x.unique())}
        unique_edge_types = {edge_type.item(): idx for idx, edge_type in enumerate(datapoint.edge_attr.unique())}
        
        datapoint.x = torch.tensor([unique_node_types[node_type.item()] for node_type in datapoint.x]).reshape(-1, 1)
        datapoint.edge_attr = torch.tensor([unique_edge_types[edge_type.item()] for edge_type in datapoint.edge_attr])
        return datapoint
    
    def deidxify(self, datapoint):
        '''
        Converts node and edge indices back to their original types
        '''
        datapoint = datapoint.clone()
        unique_node_types = {idx: node_type.item() for idx, node_type in enumerate(datapoint.x.unique())}
        unique_edge_types = {idx: edge_type.item() for idx, edge_type in enumerate(datapoint.edge_attr.unique())}
        
        datapoint.x = torch.tensor([unique_node_types.get(node_idx.item(), self.NODE_MASK) for node_idx in datapoint.x]).reshape(-1, 1)
        datapoint.edge_attr = torch.tensor([unique_edge_types.get(edge_idx.item(), self.EDGE_MASK) for edge_idx in datapoint.edge_attr])
        return datapoint

    def is_masked(self, datapoint, node=None):
        '''
        returns if node is masked or not, or array of masked nodes if node == None
        '''
        if node is None:
            return datapoint.x == self.NODE_MASK
        return datapoint.x[node] == self.NODE_MASK

    def remove_node(self, datapoint, node):
        '''
        Removes node from graph, and all edges connected to it
        '''
        assert node < datapoint.x.shape[0], "Node does not exist"
        if datapoint.x.shape[0] == 1:
            return datapoint.clone()
        datapoint = datapoint.clone()
        # remove node
        datapoint.x = torch.cat([datapoint.x[:node], datapoint.x[node+1:]])

        
        # remove edges from edge_index (remove elements containing node in tuple of edge_index)
        if datapoint.edge_index.shape[1] > 0:
            # Find edges that don't contain the node to be removed
            edge_mask = (datapoint.edge_index[0] != node) & (datapoint.edge_index[1] != node)
            
            if edge_mask.any():
                # Keep only edges that don't contain the node
                datapoint.edge_index = datapoint.edge_index[:, edge_mask]
                datapoint.edge_attr = datapoint.edge_attr[edge_mask]
                
                # update indices of edge_index
                datapoint.edge_index[datapoint.edge_index > node] -= 1
            else:
                # If no edges remain, create empty tensors
                datapoint.edge_index = torch.empty((2, 0), dtype=torch.long)
                datapoint.edge_attr = torch.empty((0,), dtype=torch.long)
        return datapoint

    def add_masked_node(self, datapoint):
        '''
        Adds a masked node to the graph
        '''
        datapoint = datapoint.clone()
        n_nodes = datapoint.x.shape[0]
        datapoint.x = torch.cat([datapoint.x.reshape(-1,1), torch.tensor([[self.NODE_MASK]])], dim=0)
        
        # Handle edge_attr properly - it should be 1D for integer edge types
        if datapoint.edge_attr.dim() == 1:
            datapoint.edge_attr = torch.cat([datapoint.edge_attr, torch.tensor([self.EDGE_MASK]).repeat(n_nodes+1)], dim=0)
        else:
            datapoint.edge_attr = torch.cat([datapoint.edge_attr.reshape(-1,1), torch.tensor([self.EDGE_MASK]).repeat(n_nodes+1, 1)], dim=0)
        
        new_edges = torch.tensor([(node, n_nodes) for node in range(n_nodes+1)], dtype=torch.long).transpose(1,0)
        datapoint.edge_index = torch.cat([datapoint.edge_index, new_edges], dim=1)
        return datapoint


    def mask_node(self, datapoint, selected_node):
        '''
        Masking node mechanism
        1. Masked node (x = -1)
        2. Connected to all other nodes in graph by masked edges (edge_attr = -1)
        
        datapoint.x: node feature matrix
        datapoint.edge_index: edge index matrix
        datapoint.edge_attr: edge attribute matrix
        datapoint.y: target value
        '''
        # mask node
        datapoint = datapoint.clone()
        datapoint.x[selected_node] = self.NODE_MASK
        
        # mask edges - handle edge cases
        if datapoint.edge_index.shape[1] > 0:
            # Find edges connected to the selected node
            edge_mask_0 = datapoint.edge_index[0] == selected_node
            edge_mask_1 = datapoint.edge_index[1] == selected_node
            
            # Ensure edge_attr has the right shape for indexing
            if datapoint.edge_attr.dim() == 1:
                datapoint.edge_attr[edge_mask_0] = self.EDGE_MASK
                datapoint.edge_attr[edge_mask_1] = self.EDGE_MASK
            else:
                # If edge_attr is multi-dimensional, we need to handle it differently
                if edge_mask_0.any():
                    datapoint.edge_attr[edge_mask_0] = self.EDGE_MASK
                if edge_mask_1.any():
                    datapoint.edge_attr[edge_mask_1] = self.EDGE_MASK
        
        return datapoint
    
    def _reorder_edge_attr_and_index(self, graph):
        '''
        Reorders edge_attr and edge_index to be like on nx graph
        (0, 0), (0, 1), (0, 2), ..., (0, n), (1, 0), (1, 1), ..., (n, n)
        '''
        graph = graph.clone()
        # reorder edge_attr
        edge_attr = torch.full((graph.x.shape[0], graph.x.shape[0]), self.EMPTY_EDGE, dtype=torch.long)
        for edge_attr_value, edge_index in zip(graph.edge_attr, graph.edge_index.T):
            edge_attr[edge_index[0], edge_index[1]] = edge_attr_value
        graph.edge_attr = edge_attr.view(-1)
        
        # reorder edge_index
        edge_index = torch.stack([torch.tensor([i, j]) for i in range(graph.x.shape[0]) for j in range(graph.x.shape[0])], dim=1)
        graph.edge_index = edge_index.long()
        return graph


    def remove_empty_edges(self, graph):
        '''
        Removes empty edges from graph
        '''
        graph = graph.clone()
        # remove masker.EMPTY_EDGE from edge_attr, and equivalent in edge_index
        if graph.edge_attr.dim() == 1:
            mask = graph.edge_attr != self.EMPTY_EDGE
        else:
            mask = graph.edge_attr.squeeze() != self.EMPTY_EDGE
        
        graph.edge_index = graph.edge_index[:, mask]
        graph.edge_attr = graph.edge_attr[mask]

        return graph

    def demask_node(self, graph, selected_node, node_type, connections_types):
        '''
        Demasking node mechanism
        1. Unmasked node (graph.x = node_type)
        2. Connected to all other nodes in graph by unmasked edges (graph.edge_attr <= connections_types)
        '''
        assert connections_types.shape[0] == graph.x.shape[0], "Number of connections must be equal to number of nodes"
        
        # demask node
        graph = graph.clone()
        graph.x[selected_node] = node_type
        
        # demask edge_attr - handle edge cases
        if graph.edge_index.shape[1] > 0:
            for i, connection in enumerate(connections_types):
                if not self.is_masked(graph, node=i):
                    # Find edges between node i and selected_node
                    edge_mask_0 = torch.logical_and(graph.edge_index[0] == i, graph.edge_index[1] == selected_node)
                    edge_mask_1 = torch.logical_and(graph.edge_index[1] == i, graph.edge_index[0] == selected_node)
                    
                    if edge_mask_0.any():
                        graph.edge_attr[edge_mask_0] = connection
                    if edge_mask_1.any():
                        graph.edge_attr[edge_mask_1] = connection
        
        return graph
    def fully_connect(self, graph, keep_original_edges=True):
        '''
        Fully connect graph with edge attribute value
        '''
        adjacency_matrix = to_dense_adj(graph.edge_index)[0]
        adjacency_matrix[adjacency_matrix == 0] = 1

        fully_connected = graph.clone()
        fully_connected.edge_attr = torch.ones(fully_connected.x.shape[0]**2) * self.EMPTY_EDGE
        
        fully_connected.edge_attr = fully_connected.edge_attr.long()

        if keep_original_edges:
            # restore values of original edges
            for edge_attr, edge_index in zip(graph.edge_attr, graph.edge_index.T):
                fully_connected.edge_attr[edge_index[0] * fully_connected.x.shape[0] + edge_index[1]] = edge_attr
                fully_connected.edge_attr[edge_index[1] * fully_connected.x.shape[0] + edge_index[0]] = edge_attr  # Ensure symmetry

        fully_connected.edge_index = torch.nonzero(adjacency_matrix).T
        return fully_connected
    
    def generate_fully_masked(self, n_nodes):
        '''
        Generates a fully masked graph like the one provided
        '''
        
        fully_masked = Data(
            x=torch.ones((n_nodes, 1))*self.NODE_MASK,
            edge_index=torch.tensor([(i, j) for i in range(n_nodes) for j in range(n_nodes)], dtype=torch.int64).transpose(0,1),
            edge_attr=torch.ones(n_nodes**2)*self.EDGE_MASK,
        )
        return fully_masked

    def get_denoised_nodes(self, graph):
        '''
        Returns a list of nodes that are denoised
        '''
        denoised_nodes = []
        for node in range(graph.x.shape[0]):
            if not self.is_masked(graph, node):
                denoised_nodes.append(node)

        return denoised_nodes