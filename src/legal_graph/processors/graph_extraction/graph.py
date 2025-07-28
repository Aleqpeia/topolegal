#!/usr/bin/env python3
"""
Legal Knowledge Graph Visualizer using NetworkX
Usage: python graph_visualizer.py results.json --output graphs/
"""

import json
import argparse
import networkx as nx
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from matplotlib.lines import Line2D
from pathlib import Path
from typing import Dict, List, Optional, Tuple
import logging
from collections import defaultdict, Counter
import seaborn as sns
import numpy as np

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

class LegalGraphVisualizer:
    def __init__(self):
        self.node_colors = {
            'PERSON': '#FF6B6B',      # Red for persons (ОСОБА_X)
            'COURT': '#4ECDC4',       # Teal for courts
            'CRIME': '#FFE66D',       # Yellow for crimes
            'LEGAL_REF': '#95E1D3',   # Light green for legal references
            'COMPANY': '#A8E6CF',     # Light blue for companies
            'LOCATION': '#FFB3BA',    # Light pink for locations
            'ROLE': '#DDA0DD',        # Light purple for roles
            'OTHER': '#D3D3D3'        # Gray for others
        }
        
        # Single edge color - focus on confidence levels through thickness
        self.edge_color = '#2E2E2E'  # Dark gray for all edges

    def load_results(self, file_path: str) -> List[Dict]:
        """Load knowledge graph results from JSON file"""
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                return json.load(f)
        except Exception as e:
            logger.error(f"Error loading {file_path}: {e}")
            return []

    def create_graph_from_document(self, doc_data: Dict, min_confidence: float = 0.0) -> nx.DiGraph:
        """Create NetworkX graph from single document data"""
        G = nx.DiGraph()
        
        if 'error' in doc_data:
            logger.warning(f"Skipping document {doc_data.get('doc_id')} due to error")
            return G
        
        kg = doc_data.get('knowledge_graph', {})
        triplets = kg.get('triplets', [])
        entities = kg.get('entities', [])
        
        # Add all entities as nodes
        for entity in entities:
            node_id = entity['text']
            node_type = self._classify_node_type(entity['text'], entity['label'])
            
            G.add_node(node_id, 
                      label=entity['label'],
                      node_type=node_type,
                      start=entity['start'],
                      end=entity['end'],
                      color=self.node_colors.get(node_type, self.node_colors['OTHER']))
        
        # Add triplets as edges (only above confidence threshold)
        for triplet in triplets:
            if triplet.get('confidence', 0) >= min_confidence:
                source = triplet['source']
                target = triplet['target']
                relation = triplet['relation']
                
                # Only add edge if both nodes exist
                if G.has_node(source) and G.has_node(target):
                    G.add_edge(source, target,
                              relation=relation,
                              legal_reference=triplet.get('legal_reference', ''),
                              confidence=triplet.get('confidence', 0),
                              weight=triplet.get('confidence', 0))
        
        # Add graph metadata
        G.graph['doc_id'] = doc_data.get('doc_id', 'unknown')
        G.graph['entities_count'] = len(entities)
        G.graph['triplets_count'] = len(triplets)
        G.graph['high_conf_edges'] = len([t for t in triplets if t.get('confidence', 0) >= min_confidence])
        
        return G

    def create_combined_graph(self, results: List[Dict], min_confidence: float = 0.6) -> nx.DiGraph:
        """Create combined graph from multiple documents"""
        combined_G = nx.DiGraph()
        
        logger.info(f"Creating combined graph from {len(results)} documents")
        
        for doc_data in results:
            if 'error' in doc_data:
                continue
                
            doc_graph = self.create_graph_from_document(doc_data, min_confidence)
            
            # Merge nodes (keep attributes from first occurrence)
            for node, attrs in doc_graph.nodes(data=True):
                if not combined_G.has_node(node):
                    combined_G.add_node(node, **attrs)
            
            # Merge edges (aggregate confidence scores)
            for source, target, attrs in doc_graph.edges(data=True):
                if combined_G.has_edge(source, target):
                    # Update confidence to maximum
                    existing_conf = combined_G[source][target].get('confidence', 0)
                    new_conf = max(existing_conf, attrs.get('confidence', 0))
                    combined_G[source][target]['confidence'] = new_conf
                    combined_G[source][target]['weight'] = new_conf
                    
                    # Combine legal references
                    existing_refs = combined_G[source][target].get('legal_reference', '')
                    new_refs = attrs.get('legal_reference', '')
                    if new_refs and new_refs not in existing_refs:
                        combined_refs = f"{existing_refs}; {new_refs}" if existing_refs else new_refs
                        combined_G[source][target]['legal_reference'] = combined_refs
                else:
                    combined_G.add_edge(source, target, **attrs)
        
        combined_G.graph['type'] = 'combined'
        combined_G.graph['num_documents'] = len([d for d in results if 'error' not in d])
        combined_G.graph['min_confidence'] = min_confidence
        
        logger.info(f"Combined graph: {combined_G.number_of_nodes()} nodes, {combined_G.number_of_edges()} edges")
        return combined_G

    def visualize_graph(self, G: nx.DiGraph, output_path: str, layout: str = 'spring', 
                       figsize: Tuple[int, int] = (15, 10), show_labels: bool = True):
        """Visualize graph with legal domain styling"""
        
        if G.number_of_nodes() == 0:
            logger.warning("Empty graph, skipping visualization")
            return
        
        plt.figure(figsize=figsize)
        
        # Choose layout
        if layout == 'spring':
            pos = nx.spring_layout(G, k=2, iterations=50)
        elif layout == 'circular':
            pos = nx.circular_layout(G)
        elif layout == 'hierarchical':
            pos = self._hierarchical_layout(G)
        else:
            pos = nx.spring_layout(G)
        
        # Draw nodes by type
        node_types = set(nx.get_node_attributes(G, 'node_type').values())
        
        for node_type in node_types:
            nodes = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == node_type]
            if nodes:
                color = self.node_colors.get(node_type, self.node_colors['OTHER'])
                nx.draw_networkx_nodes(G, pos, nodelist=nodes, 
                                     node_color=color, node_size=1000, alpha=0.8)
        
        # Draw edges by confidence level only - consistent colors
        edges_high_conf = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('confidence', 0) >= 0.8]
        edges_med_conf = [(u, v) for u, v, attrs in G.edges(data=True) if 0.6 <= attrs.get('confidence', 0) < 0.8]
        edges_low_conf = [(u, v) for u, v, attrs in G.edges(data=True) if attrs.get('confidence', 0) < 0.6]
        
        # High confidence edges - thick, dark
        if edges_high_conf:
            nx.draw_networkx_edges(G, pos, edgelist=edges_high_conf, 
                                 edge_color=self.edge_color, width=3, alpha=0.9, 
                                 arrows=True, arrowsize=20, arrowstyle='->')
        
        # Medium confidence edges - medium thickness
        if edges_med_conf:
            nx.draw_networkx_edges(G, pos, edgelist=edges_med_conf, 
                                 edge_color=self.edge_color, width=2, alpha=0.7, 
                                 arrows=True, arrowsize=15, arrowstyle='->')
        
        # Low confidence edges - thin, light
        if edges_low_conf:
            nx.draw_networkx_edges(G, pos, edgelist=edges_low_conf, 
                                 edge_color=self.edge_color, width=1, alpha=0.5, 
                                 arrows=True, arrowsize=10, arrowstyle='->')
        
        # Add labels
        if show_labels:
            # Shorten long labels
            labels = {}
            for node in G.nodes():
                if len(node) > 15:
                    labels[node] = node[:12] + "..."
                else:
                    labels[node] = node
            
            nx.draw_networkx_labels(G, pos, labels, font_size=8, font_weight='bold')
        
        # Add title and legend
        doc_id = G.graph.get('doc_id', 'Combined')
        plt.title(f'Legal Knowledge Graph - Document {doc_id}', size=16, fontweight='bold')
        
        # Create legends
        # Node type legend
        node_legend_elements = []
        for node_type, color in self.node_colors.items():
            if any(attrs.get('node_type') == node_type for _, attrs in G.nodes(data=True)):
                node_legend_elements.append(patches.Patch(color=color, label=node_type))
        
        # Edge confidence legend  
        edge_legend_elements = [
            Line2D([0], [0], color=self.edge_color, linewidth=3, alpha=0.9, label='High confidence (0.8+)'),
            Line2D([0], [0], color=self.edge_color, linewidth=2, alpha=0.7, label='Medium confidence (0.6-0.8)'),
            Line2D([0], [0], color=self.edge_color, linewidth=1, alpha=0.5, label='Low confidence (<0.6)')
        ]
        
        # Add both legends
        if node_legend_elements:
            legend1 = plt.legend(handles=node_legend_elements, loc='upper left', bbox_to_anchor=(1, 1), title="Node Types")
            plt.gca().add_artist(legend1)
        
        if edge_legend_elements:
            plt.legend(handles=edge_legend_elements, loc='upper left', bbox_to_anchor=(1, 0.7), title="Edge Confidence")
        
        plt.axis('off')
        plt.tight_layout()
        
        # Save
        plt.savefig(output_path, dpi=300, bbox_inches='tight')
        logger.info(f"Graph saved to {output_path}")
        plt.close()

    def create_subgraphs_by_person(self, G: nx.DiGraph, output_dir: str):
        """Create separate subgraphs for each person (ОСОБА_X)"""
        persons = [n for n in G.nodes() if n.startswith('ОСОБА_')]
        
        logger.info(f"Creating subgraphs for {len(persons)} persons")
        
        for person in persons:
            # Get ego graph (person + immediate neighbors)
            if G.has_node(person):
                ego_graph = nx.ego_graph(G, person, radius=2)
                
                if ego_graph.number_of_edges() > 0:
                    output_path = Path(output_dir) / f"person_{person}_graph.png"
                    ego_graph.graph['doc_id'] = f"Person {person}"
                    self.visualize_graph(ego_graph, str(output_path), figsize=(12, 8))

    def export_graph_data(self, G: nx.DiGraph, output_path: str):
        """Export graph in various formats"""
        base_path = Path(output_path).with_suffix('')
        
        # Export as GraphML (preserves all attributes)
        nx.write_graphml(G, f"{base_path}.graphml")
        
        # Export as GML (simple format)
        nx.write_gml(G, f"{base_path}.gml")
        
        # Export edge list with attributes
        with open(f"{base_path}_edges.csv", 'w', encoding='utf-8') as f:
            f.write("source,target,relation,legal_reference,confidence\n")
            for source, target, attrs in G.edges(data=True):
                relation = attrs.get('relation', '')
                legal_ref = attrs.get('legal_reference', '').replace(',', ';')
                confidence = attrs.get('confidence', 0)
                f.write(f'"{source}","{target}","{relation}","{legal_ref}",{confidence}\n')
        
        # Export node list with attributes
        with open(f"{base_path}_nodes.csv", 'w', encoding='utf-8') as f:
            f.write("node,label,node_type\n")
            for node, attrs in G.nodes(data=True):
                label = attrs.get('label', '')
                node_type = attrs.get('node_type', '')
                f.write(f'"{node}","{label}","{node_type}"\n')
        
        logger.info(f"Graph data exported to {base_path}.*")

    def generate_graph_statistics(self, G: nx.DiGraph) -> Dict:
        """Generate comprehensive graph statistics"""
        stats = {
            'basic_stats': {
                'nodes': G.number_of_nodes(),
                'edges': G.number_of_edges(),
                'density': nx.density(G),
                'is_connected': nx.is_weakly_connected(G)
            },
            'node_types': dict(Counter(attrs.get('node_type', 'OTHER') 
                                     for _, attrs in G.nodes(data=True))),
            'confidence_distribution': {
                'high (0.8+)': len([(u, v) for u, v, attrs in G.edges(data=True) 
                                   if attrs.get('confidence', 0) >= 0.8]),
                'medium (0.6-0.8)': len([(u, v) for u, v, attrs in G.edges(data=True) 
                                        if 0.6 <= attrs.get('confidence', 0) < 0.8]),
                'low (0.4-0.6)': len([(u, v) for u, v, attrs in G.edges(data=True) 
                                     if 0.4 <= attrs.get('confidence', 0) < 0.6]),
                'very_low (<0.4)': len([(u, v) for u, v, attrs in G.edges(data=True) 
                                       if attrs.get('confidence', 0) < 0.4])
            },
            'top_relations': dict(Counter(attrs.get('relation', '') 
                                        for _, _, attrs in G.edges(data=True)).most_common(10)),
            'central_nodes': {
                'degree': dict(sorted(list(G.degree()), key=lambda x: x[1], reverse=True)[:10]),
                'in_degree': dict(sorted(list(G.in_degree()), key=lambda x: x[1], reverse=True)[:10]),
                'out_degree': dict(sorted(list(G.out_degree()), key=lambda x: x[1], reverse=True)[:10])
            }
        }
        
        # Add connected components info
        if not nx.is_weakly_connected(G):
            components = list(nx.weakly_connected_components(G))
            stats['components'] = {
                'count': len(components),
                'largest_size': max(len(c) for c in components) if components else 0,
                'sizes': [len(c) for c in components]
            }
        
        return stats

    def _classify_node_type(self, text: str, label: str) -> str:
        """Classify node type for visualization"""
        text_lower = text.lower()
        
        if text.startswith('ОСОБА_'):
            return 'PERSON'
        elif 'суд' in text_lower:
            return 'COURT'
        elif label == 'CRIME' or 'крадіжка' in text_lower:
            return 'CRIME'
        elif any(ref in text_lower for ref in ['ст.', 'ч.', 'кк', 'кпк']):
            return 'LEGAL_REF'
        elif label in ['ORG'] and any(company in text_lower for company in ['тов', 'ооо', 'пп']):
            return 'COMPANY'
        elif label in ['LOC', 'LOCATION']:
            return 'LOCATION'
        elif label == 'ROLE':
            return 'ROLE'
        else:
            return 'OTHER'

    def _hierarchical_layout(self, G: nx.DiGraph) -> Dict:
        """Create hierarchical layout with courts at top, persons in middle, crimes at bottom"""
        pos = {}
        
        # Classify nodes by type
        courts = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == 'COURT']
        persons = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == 'PERSON']
        crimes = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') == 'CRIME']
        others = [n for n, attrs in G.nodes(data=True) if attrs.get('node_type') not in ['COURT', 'PERSON', 'CRIME']]
        
        # Position nodes in layers
        y_positions = [3, 2, 1, 0]  # courts, persons, crimes, others
        node_groups = [courts, persons, crimes, others]
        
        for y_pos, group in zip(y_positions, node_groups):
            if group:
                for i, node in enumerate(group):
                    x_pos = (i - len(group)/2) * 2
                    pos[node] = (x_pos, y_pos)
        
        return pos

def main():
    parser = argparse.ArgumentParser(description="Visualize legal knowledge graphs")
    parser.add_argument("input_file", help="JSON file with knowledge graph results")
    parser.add_argument("--output", "-o", default="graphs/", help="Output directory")
    parser.add_argument("--min-confidence", "-c", type=float, default=0.6, 
                       help="Minimum confidence for edges")
    parser.add_argument("--layout", choices=['spring', 'circular', 'hierarchical'], 
                       default='spring', help="Graph layout")
    parser.add_argument("--combined", action="store_true", 
                       help="Create combined graph from all documents")
    parser.add_argument("--individual", action="store_true", 
                       help="Create individual graphs for each document")
    parser.add_argument("--subgraphs", action="store_true", 
                       help="Create subgraphs for each person")
    parser.add_argument("--export", action="store_true", 
                       help="Export graph data in multiple formats")
    parser.add_argument("--stats", action="store_true", 
                       help="Generate graph statistics")
    
    args = parser.parse_args()
    
    # Create output directory
    output_dir = Path(args.output)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Load results
    visualizer = LegalGraphVisualizer()
    results = visualizer.load_results(args.input_file)
    
    if not results:
        logger.error("No valid results to process")
        return
    
    logger.info(f"Loaded {len(results)} documents")
    
    # Create combined graph
    if args.combined or not (args.individual or args.subgraphs):
        logger.info("Creating combined graph...")
        combined_graph = visualizer.create_combined_graph(results, args.min_confidence)
        
        if combined_graph.number_of_nodes() > 0:
            # Visualize
            output_path = output_dir / f"combined_graph_conf_{args.min_confidence}.png"
            visualizer.visualize_graph(combined_graph, str(output_path), args.layout)
            
            # Export data
            if args.export:
                export_path = output_dir / f"combined_graph_conf_{args.min_confidence}"
                visualizer.export_graph_data(combined_graph, str(export_path))
            
            # Generate stats
            if args.stats:
                stats = visualizer.generate_graph_statistics(combined_graph)
                stats_path = output_dir / f"combined_graph_stats_conf_{args.min_confidence}.json"
                with open(stats_path, 'w', encoding='utf-8') as f:
                    json.dump(stats, f, indent=2, ensure_ascii=False)
                logger.info(f"Statistics saved to {stats_path}")
            
            # Create person subgraphs
            if args.subgraphs:
                subgraph_dir = output_dir / "person_subgraphs"
                subgraph_dir.mkdir(exist_ok=True)
                visualizer.create_subgraphs_by_person(combined_graph, str(subgraph_dir))
        
    # Create individual graphs
    if args.individual:
        logger.info("Creating individual document graphs...")
        individual_dir = output_dir / "individual_docs"
        individual_dir.mkdir(exist_ok=True)
        
        for doc_data in results:
            if 'error' not in doc_data:
                doc_id = doc_data.get('doc_id', 'unknown')
                doc_graph = visualizer.create_graph_from_document(doc_data, args.min_confidence)
                
                if doc_graph.number_of_edges() > 0:
                    output_path = individual_dir / f"doc_{doc_id}_graph.png"
                    visualizer.visualize_graph(doc_graph, str(output_path), args.layout, figsize=(12, 8))

if __name__ == "__main__":
    # Required packages check
    try:
        import matplotlib.pyplot as plt
        import seaborn as sns
    except ImportError:
        print("Required packages missing. Install with:")
        print("pip install matplotlib seaborn networkx")
        exit(1)
    
    main()