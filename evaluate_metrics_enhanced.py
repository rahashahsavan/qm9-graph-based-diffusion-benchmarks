#!/usr/bin/env python3
"""
Enhanced evaluation script for molecules with both graph and SMILES data.

This enhanced version can utilize both graph representations and SMILES strings
for more comprehensive evaluation of generated molecules.

Usage:
    python evaluate_metrics_enhanced.py --generated_graphs graphs.pkl --generated_smiles smiles.smi --reference_smiles reference.smi
"""

import os
import sys
import json
import pickle
import argparse
import numpy as np
import pandas as pd
from typing import List, Dict, Any, Tuple, Optional, Union
from collections import Counter
import warnings
warnings.filterwarnings('ignore')

# Core dependencies
try:
    import torch
    import torch.nn.functional as F
    TORCH_AVAILABLE = True
except ImportError:
    print("Warning: PyTorch not available. Some metrics will be skipped.")
    TORCH_AVAILABLE = False

try:
    from rdkit import Chem
    from rdkit.Chem import Descriptors, Crippen, QED, rdMolDescriptors
    RDKIT_AVAILABLE = True
except ImportError:
    print("Warning: RDKit not available. Some metrics will be skipped.")
    RDKIT_AVAILABLE = False

try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    print("Warning: NetworkX not available. Graph metrics will be skipped.")
    NETWORKX_AVAILABLE = False

try:
    from sklearn.metrics import pairwise_distances
    from sklearn.preprocessing import StandardScaler
    SKLEARN_AVAILABLE = True
except ImportError:
    print("Warning: scikit-learn not available. Some metrics will be skipped.")
    SKLEARN_AVAILABLE = False

# Import the original evaluation functions
try:
    from evaluate_metrics import (
        load_smiles, compute_validity, compute_uniqueness, compute_novelty,
        compute_fcd, compute_atom_stability, compute_mol_stability,
        compute_mmd, compute_nll, save_results, print_results
    )
    ORIGINAL_METRICS_AVAILABLE = True
except ImportError:
    print("Warning: Original evaluate_metrics.py not found. Some metrics will be skipped.")
    ORIGINAL_METRICS_AVAILABLE = False


def load_graph_data(graph_file: str) -> List[Tuple]:
    """Load graph data from pickle file."""
    print(f"Loading graphs from: {graph_file}")
    
    if not os.path.exists(graph_file):
        raise FileNotFoundError(f"Graph file not found: {graph_file}")
    
    with open(graph_file, 'rb') as f:
        graphs = pickle.load(f)
    
    print(f"Loaded {len(graphs)} graphs")
    return graphs


def graph_to_networkx(graph_tuple: Tuple) -> nx.Graph:
    """Convert graph tuple (X, E) to NetworkX graph."""
    if not NETWORKX_AVAILABLE:
        return None
    
    try:
        X, E = graph_tuple
        if isinstance(X, torch.Tensor):
            X = X.cpu().numpy()
        if isinstance(E, torch.Tensor):
            E = E.cpu().numpy()
        
        # Create adjacency matrix from edge features
        n_nodes = X.shape[0]
        adj_matrix = np.zeros((n_nodes, n_nodes))
        
        # Convert edge features to adjacency matrix
        if len(E.shape) == 3:  # Edge features format
            adj_matrix = E.sum(axis=2)  # Sum over edge types
        elif len(E.shape) == 2:  # Simple adjacency matrix
            adj_matrix = E
        
        # Create NetworkX graph
        G = nx.from_numpy_array(adj_matrix)
        
        # Add node attributes (atom types)
        for i, atom_type in enumerate(X):
            G.nodes[i]['atom_type'] = int(atom_type)
        
        return G
    except Exception as e:
        print(f"Error converting graph to NetworkX: {e}")
        return None


def compute_graph_topology_metrics(generated_graphs: List[Tuple], reference_graphs: List[Tuple] = None) -> Dict[str, Any]:
    """Compute graph topology-based metrics."""
    print("Computing graph topology metrics...")
    
    if not NETWORKX_AVAILABLE:
        return {'graph_metrics': None}
    
    try:
        # Convert graphs to NetworkX format
        gen_nx_graphs = []
        ref_nx_graphs = []
        
        for graph in generated_graphs:
            nx_graph = graph_to_networkx(graph)
            if nx_graph is not None:
                gen_nx_graphs.append(nx_graph)
        
        if reference_graphs:
            for graph in reference_graphs:
                nx_graph = graph_to_networkx(graph)
                if nx_graph is not None:
                    ref_nx_graphs.append(nx_graph)
        
        if not gen_nx_graphs:
            return {'graph_metrics': None}
        
        # Compute graph statistics
        gen_stats = compute_graph_statistics(gen_nx_graphs)
        
        metrics = {
            'graph_metrics': {
                'generated_stats': gen_stats,
                'generated_count': len(gen_nx_graphs)
            }
        }
        
        if ref_nx_graphs:
            ref_stats = compute_graph_statistics(ref_nx_graphs)
            metrics['graph_metrics']['reference_stats'] = ref_stats
            metrics['graph_metrics']['reference_count'] = len(ref_nx_graphs)
            
            # Compute distribution differences
            distribution_diff = compute_distribution_differences(gen_stats, ref_stats)
            metrics['graph_metrics']['distribution_differences'] = distribution_diff
        
        return metrics
    
    except Exception as e:
        print(f"Error computing graph topology metrics: {e}")
        return {'graph_metrics': None}


def compute_graph_statistics(graphs: List[nx.Graph]) -> Dict[str, Any]:
    """Compute statistical properties of graph collection."""
    if not graphs:
        return {}
    
    stats = {
        'num_nodes': [],
        'num_edges': [],
        'density': [],
        'avg_clustering': [],
        'avg_shortest_path': [],
        'degree_centrality': [],
        'betweenness_centrality': [],
        'eigenvector_centrality': []
    }
    
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
            
        stats['num_nodes'].append(G.number_of_nodes())
        stats['num_edges'].append(G.number_of_edges())
        stats['density'].append(nx.density(G))
        
        if G.number_of_nodes() > 1:
            stats['avg_clustering'].append(nx.average_clustering(G))
            
            if nx.is_connected(G):
                stats['avg_shortest_path'].append(nx.average_shortest_path_length(G))
            
            # Centrality measures
            degree_cent = nx.degree_centrality(G)
            stats['degree_centrality'].extend(list(degree_cent.values()))
            
            betweenness_cent = nx.betweenness_centrality(G)
            stats['betweenness_centrality'].extend(list(betweenness_cent.values()))
            
            try:
                eigenvector_cent = nx.eigenvector_centrality(G)
                stats['eigenvector_centrality'].extend(list(eigenvector_cent.values()))
            except:
                pass
    
    # Compute summary statistics
    summary = {}
    for key, values in stats.items():
        if values:
            summary[key] = {
                'mean': np.mean(values),
                'std': np.std(values),
                'min': np.min(values),
                'max': np.max(values),
                'median': np.median(values)
            }
        else:
            summary[key] = None
    
    return summary


def compute_distribution_differences(gen_stats: Dict, ref_stats: Dict) -> Dict[str, float]:
    """Compute differences between generated and reference distributions."""
    differences = {}
    
    for metric in ['num_nodes', 'num_edges', 'density', 'avg_clustering']:
        if (metric in gen_stats and gen_stats[metric] is not None and 
            metric in ref_stats and ref_stats[metric] is not None):
            
            gen_mean = gen_stats[metric]['mean']
            ref_mean = ref_stats[metric]['mean']
            
            # Relative difference
            if ref_mean != 0:
                differences[f'{metric}_relative_diff'] = abs(gen_mean - ref_mean) / ref_mean
            else:
                differences[f'{metric}_relative_diff'] = float('inf')
            
            # Absolute difference
            differences[f'{metric}_absolute_diff'] = abs(gen_mean - ref_mean)
    
    return differences


def compute_enhanced_nspdk(generated_graphs: List[Tuple], reference_graphs: List[Tuple] = None) -> Dict[str, Any]:
    """Enhanced NSPDK using actual graph representations."""
    print("Computing enhanced NSPDK...")
    
    if not NETWORKX_AVAILABLE or not SKLEARN_AVAILABLE:
        return {'enhanced_nspdk': None}
    
    try:
        # Convert to NetworkX graphs
        gen_nx_graphs = [graph_to_networkx(g) for g in generated_graphs]
        gen_nx_graphs = [g for g in gen_nx_graphs if g is not None]
        
        if not gen_nx_graphs:
            return {'enhanced_nspdk': None}
        
        # Extract graph features
        gen_features = extract_graph_features(gen_nx_graphs)
        
        metrics = {
            'enhanced_nspdk': {
                'generated_features': gen_features,
                'generated_count': len(gen_nx_graphs)
            }
        }
        
        if reference_graphs:
            ref_nx_graphs = [graph_to_networkx(g) for g in reference_graphs]
            ref_nx_graphs = [g for g in ref_nx_graphs if g is not None]
            
            if ref_nx_graphs:
                ref_features = extract_graph_features(ref_nx_graphs)
                metrics['enhanced_nspdk']['reference_features'] = ref_features
                metrics['enhanced_nspdk']['reference_count'] = len(ref_nx_graphs)
                
                # Compute kernel similarity
                similarity = compute_graph_kernel_similarity(gen_features, ref_features)
                metrics['enhanced_nspdk']['kernel_similarity'] = similarity
        
        return metrics
    
    except Exception as e:
        print(f"Error computing enhanced NSPDK: {e}")
        return {'enhanced_nspdk': None}


def extract_graph_features(graphs: List[nx.Graph]) -> np.ndarray:
    """Extract comprehensive features from graphs."""
    features = []
    
    for G in graphs:
        if G.number_of_nodes() == 0:
            continue
        
        feature_vector = []
        
        # Basic graph properties
        feature_vector.extend([
            G.number_of_nodes(),
            G.number_of_edges(),
            nx.density(G),
            nx.number_of_connected_components(G)
        ])
        
        # Clustering and path properties
        if G.number_of_nodes() > 1:
            feature_vector.append(nx.average_clustering(G))
            if nx.is_connected(G):
                feature_vector.append(nx.average_shortest_path_length(G))
            else:
                feature_vector.append(0)
        else:
            feature_vector.extend([0, 0])
        
        # Degree statistics
        degrees = [d for n, d in G.degree()]
        if degrees:
            feature_vector.extend([
                np.mean(degrees),
                np.std(degrees),
                np.min(degrees),
                np.max(degrees)
            ])
        else:
            feature_vector.extend([0, 0, 0, 0])
        
        # Centrality measures
        if G.number_of_nodes() > 0:
            degree_cent = nx.degree_centrality(G)
            feature_vector.append(np.mean(list(degree_cent.values())))
            
            betweenness_cent = nx.betweenness_centrality(G)
            feature_vector.append(np.mean(list(betweenness_cent.values())))
        else:
            feature_vector.extend([0, 0])
        
        # Atom type distribution (if available)
        atom_types = [G.nodes[n].get('atom_type', 0) for n in G.nodes()]
        atom_type_counts = Counter(atom_types)
        for atom_type in range(10):  # Common atom types
            feature_vector.append(atom_type_counts.get(atom_type, 0))
        
        features.append(feature_vector)
    
    return np.array(features) if features else np.array([])


def compute_graph_kernel_similarity(features1: np.ndarray, features2: np.ndarray) -> float:
    """Compute kernel-based similarity between feature sets."""
    if len(features1) == 0 or len(features2) == 0:
        return 0.0
    
    # Standardize features
    scaler = StandardScaler()
    all_features = np.vstack([features1, features2])
    all_features_scaled = scaler.fit_transform(all_features)
    
    features1_scaled = all_features_scaled[:len(features1)]
    features2_scaled = all_features_scaled[len(features1):]
    
    # Compute RBF kernel
    gamma = 1.0 / features1_scaled.shape[1]
    pairwise_dists = pairwise_distances(features1_scaled, features2_scaled, metric='euclidean')
    kernel_matrix = np.exp(-gamma * pairwise_dists**2)
    
    return np.mean(kernel_matrix)


def compute_graph_validity(generated_graphs: List[Tuple]) -> Dict[str, Any]:
    """Compute validity metrics based on graph structure."""
    print("Computing graph-based validity...")
    
    if not NETWORKX_AVAILABLE:
        return {'graph_validity': None}
    
    try:
        valid_graphs = 0
        total_graphs = len(generated_graphs)
        
        for graph in generated_graphs:
            nx_graph = graph_to_networkx(graph)
            if nx_graph is not None and nx_graph.number_of_nodes() > 0:
                # Check for reasonable graph properties
                if (nx_graph.number_of_nodes() >= 3 and  # Minimum size
                    nx_graph.number_of_nodes() <= 50 and  # Maximum size
                    nx.number_of_connected_components(nx_graph) == 1):  # Connected
                    valid_graphs += 1
        
        graph_validity = valid_graphs / total_graphs if total_graphs > 0 else 0
        
        return {
            'graph_validity': {
                'validity': graph_validity,
                'valid_count': valid_graphs,
                'total_count': total_graphs
            }
        }
    
    except Exception as e:
        print(f"Error computing graph validity: {e}")
        return {'graph_validity': None}


def main():
    parser = argparse.ArgumentParser(description='Enhanced evaluation with graph and SMILES data')
    parser.add_argument('--generated_graphs', type=str, required=True,
                       help='Path to generated graphs file (.pkl)')
    parser.add_argument('--generated_smiles', type=str, required=True,
                       help='Path to generated SMILES file (.smi, .txt, or .csv)')
    parser.add_argument('--reference_smiles', type=str, required=True,
                       help='Path to reference SMILES file (.smi, .txt, or .csv)')
    parser.add_argument('--reference_graphs', type=str, default=None,
                       help='Path to reference graphs file (.pkl) - optional')
    parser.add_argument('--output_prefix', type=str, default='enhanced_metrics_results',
                       help='Output file prefix (default: enhanced_metrics_results)')
    parser.add_argument('--max_samples', type=int, default=None,
                       help='Maximum number of samples to evaluate (for testing)')
    
    args = parser.parse_args()
    
    # Load data
    try:
        generated_graphs = load_graph_data(args.generated_graphs)
        generated_smiles = load_smiles(args.generated_smiles)
        reference_smiles = load_smiles(args.reference_smiles)
        
        reference_graphs = None
        if args.reference_graphs:
            reference_graphs = load_graph_data(args.reference_graphs)
    
    except Exception as e:
        print(f"Error loading data: {e}")
        return
    
    # Limit samples if specified
    if args.max_samples:
        generated_graphs = generated_graphs[:args.max_samples]
        generated_smiles = generated_smiles[:args.max_samples]
        reference_smiles = reference_smiles[:args.max_samples]
        if reference_graphs:
            reference_graphs = reference_graphs[:args.max_samples]
        print(f"Limited to {args.max_samples} samples for evaluation")
    
    print(f"\nEvaluating {len(generated_graphs)} generated molecules against {len(reference_smiles)} reference molecules")
    
    # Compute all metrics
    metrics = {}
    
    # Original SMILES-based metrics (if available)
    if ORIGINAL_METRICS_AVAILABLE:
        print("\nComputing SMILES-based metrics...")
        metrics['validity'] = compute_validity(generated_smiles)
        metrics['uniqueness'] = compute_uniqueness(generated_smiles)
        metrics['novelty'] = compute_novelty(generated_smiles, reference_smiles)
        metrics['atom_stability'] = compute_atom_stability(generated_smiles)
        metrics['mol_stability'] = compute_mol_stability(generated_smiles)
        metrics['fcd'] = compute_fcd(generated_smiles, reference_smiles)
        metrics['mmd'] = compute_mmd(generated_smiles, reference_smiles)
        metrics['nll'] = compute_nll(generated_smiles)
        metrics['nspdk'] = compute_nspdk(generated_smiles, reference_smiles)
    
    # Enhanced graph-based metrics
    print("\nComputing enhanced graph-based metrics...")
    metrics['graph_validity'] = compute_graph_validity(generated_graphs)
    metrics['graph_topology'] = compute_graph_topology_metrics(generated_graphs, reference_graphs)
    metrics['enhanced_nspdk'] = compute_enhanced_nspdk(generated_graphs, reference_graphs)
    
    # Save and print results
    save_results(metrics, args.output_prefix)
    
    # Enhanced printing
    print("\n" + "="*70)
    print("ENHANCED MOLECULAR GENERATION EVALUATION RESULTS")
    print("="*70)
    
    # SMILES-based metrics
    if ORIGINAL_METRICS_AVAILABLE:
        print("\n📊 SMILES-BASED METRICS:")
        print("-" * 40)
        validity = metrics.get('validity', {})
        print(f"Validity:           {validity.get('validity', 'N/A'):.4f}")
        uniqueness = metrics.get('uniqueness', {})
        print(f"Uniqueness:         {uniqueness.get('uniqueness', 'N/A'):.4f}")
        novelty = metrics.get('novelty', {})
        print(f"Novelty:            {novelty.get('novelty', 'N/A'):.4f}")
        fcd = metrics.get('fcd', {})
        print(f"FCD:                {fcd.get('fcd', 'N/A'):.4f}")
    
    # Graph-based metrics
    print("\n🔬 GRAPH-BASED METRICS:")
    print("-" * 40)
    graph_validity = metrics.get('graph_validity', {})
    if graph_validity and 'graph_validity' in graph_validity:
        gv = graph_validity['graph_validity']
        print(f"Graph Validity:     {gv.get('validity', 'N/A'):.4f}")
    
    graph_topology = metrics.get('graph_topology', {})
    if graph_topology and 'graph_metrics' in graph_topology:
        gt = graph_topology['graph_metrics']
        print(f"Generated Graphs:   {gt.get('generated_count', 'N/A')}")
        if 'reference_count' in gt:
            print(f"Reference Graphs:   {gt.get('reference_count', 'N/A')}")
    
    enhanced_nspdk = metrics.get('enhanced_nspdk', {})
    if enhanced_nspdk and 'enhanced_nspdk' in enhanced_nspdk:
        enspdk = enhanced_nspdk['enhanced_nspdk']
        print(f"Enhanced NSPDK:      {enspdk.get('kernel_similarity', 'N/A'):.4f}")
    
    print("\n" + "="*70)


if __name__ == '__main__':
    main()
