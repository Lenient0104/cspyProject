import networkx as nx
import numpy as np
from cspy import PSOLGENT
from cspy import GreedyElim
from networkx import DiGraph
from optimization import Optimization
from user_info import User
from cspy import BiDirectional
from numpy import zeros, ones, array

net_xml_path = 'DCC.net.xml'
source_edge = '361450282'
target_edge = "-110407380#1"
start_mode = 'walking'
user = User(60, True, 0, 20)
db_path = 'test_new.db'


optimizer_interface = Optimization(net_xml_path, user, db_path, source_edge, target_edge)
graph = optimizer_interface.new_graph
print(graph)


def calculate_energy_comsumption(current_mode, distance):
    if current_mode == 'walking':
        return 0
    # Define vehicle efficiency in Wh per meter (converted from Wh per km)
    vehicle_efficiency = {'e_bike_1': 20 / 1000, 'e_scooter_1': 25 / 1000, 'e_car': 150 / 1000}
    # battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 50000}
    battery_capacity = {'e_bike_1': 500, 'e_scooter_1': 250, 'e_car': 5000}
    energy_consumed = vehicle_efficiency[current_mode] * distance
    # Calculate the delta SoC (%) for the distance traveled
    delta_soc = (energy_consumed / battery_capacity[current_mode]) * 100

    return delta_soc


def build_multimodal_graph(paths_graph, start_edge, destination_edge):
    multimodal_graph = DiGraph()

    print("Starting to build the multimodal graph...")


    for source, target, data in paths_graph.edges(data=True):
        mode = data['mode']
        total_time = data['weight']
        distance = data['distance']


        virtual_node = f"{source}_{target}_{mode}"
        multimodal_graph.add_edge(source, virtual_node, weight=total_time, mode=mode, distance=distance)
        multimodal_graph.add_edge(virtual_node, target, weight=0.000, mode=mode)


    print(f"Processing walking paths from {start_edge}...")
    for target, data in paths_graph[start_edge].items():
        if 'mode' in data and data['mode'] == 'walking':
            multimodal_graph.add_edge(start_edge, target, weight=data['weight'], mode='walking', distance=data['distance'])


    print(f"Processing walking paths to {destination_edge}...")
    for source, target, data in paths_graph.in_edges(destination_edge, data=True):
        if data['mode'] == 'walking':
            multimodal_graph.add_edge(source, destination_edge, weight=data['weight'], mode='walking', distance=data['distance'])

    return multimodal_graph


def build_energy_cost_graph(multimodal_graph, source_edge, target_edge):
    G = nx.DiGraph(directed=True, n_res=2)  # For each mode, the electricity is the only constraint currently
    print("Building energy cost graph...")
    for source, target, data in multimodal_graph.edges(data=True):
        mode = data.get('mode')
        time_cost = data.get('weight')

        if 'virtual' in source or 'virtual' in target:
            energy_cost = 0
        else:
            distance = data.get('distance', 0)
            energy_cost = calculate_energy_comsumption(mode, distance) if distance > 0 else 0

        mapped_source = 'Source' if source == source_edge else source
        mapped_target = 'Sink' if target == target_edge else target

        G.add_edge(mapped_source, mapped_target, res_cost=np.array([energy_cost, 0]), weight=time_cost)
    print("end")
    return G


if __name__ == '__main__':
    # print(original_graph)
    multimodal_graph = build_multimodal_graph(graph, source_edge, target_edge)
    # print(multimodal_graph)
    G = build_energy_cost_graph(multimodal_graph, source_edge, target_edge)
    # print(G)

    # n_nodes = len(G.nodes())
    # psolgent = PSOLGENT(G, [100], [0], max_iter=100, swarm_size=50, member_size=n_nodes, neighbourhood_size=50)
    # psolgent.run()

    max_res, min_res = [100, 0], [0, 0]
    greedelim = GreedyElim(G, max_res, min_res)
    greedelim.run()
    print(greedelim.path)


    # total_time_cost = 0  # Initialize the total time cost
    #
    # print("Optimal path:")
    # path = bidirec.path
    # for i in range(len(path) - 1):
    #     source = path[i]
    #     target = path[i + 1]
    #
    #     # Skip virtual nodes when calculating total time cost
    #     if 'virtual' not in source and 'virtual' not in target:
    #         # Get the actual source and target names for the edge in the original graph
    #         actual_source = source_edge if source == 'Source' else source
    #         actual_target = target_edge if target == 'Sink' else target
    #
    #         # Find the edge data in the original graph to get mode and time cost
    #         data = graph.get_edge_data(actual_source, actual_target)
    #         mode = data['mode'] if 'mode' in data else 'walking'  # Default to walking if no mode is specified
    #         time_cost = data['weight']
    #
    #         total_time_cost += time_cost  # Accumulate the total time cost
    #
    #         print(f"From {actual_source} to {actual_target}: Mode={mode}, Time Cost={time_cost}")
    #
    # print(f"Total time cost: {total_time_cost}")
