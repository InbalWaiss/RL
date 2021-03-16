import matplotlib.pyplot as plt
import networkx as nx
from Arena.Position import Position
from Arena.CState import State
from Arena.AbsDecisionMaker import AbsDecisionMaker
from Arena.constants import *
import numpy as np

PRINT_FLAG = False


class Greedy_player(AbsDecisionMaker):

    def __init__(self, UPDATE_CONTEXT=True , path_model_to_load=None):

        self._action = -1
        self._type = AgentType.Greedy
        self.episode_number = 0
        self._epsilon = 0
        self.path_model_to_load = None

        self.all_pairs_distances = {}
        self.all_pairs_shortest_path = {}
        self.G = self.create_graph()



    def create_graph(self):
        G = nx.grid_2d_graph(SIZE_X, SIZE_Y)
        pos = dict((n, n) for n in G.nodes())  # Dictionary of all positions
        labels = dict(((i, j), (i, j)) for i, j in G.nodes())

        if NUMBER_OF_ACTIONS >= 8:
            Diagonals_Weight = 1
            # add diagonals edges
            G.add_edges_from([
                                 ((x, y), (x + 1, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ] + [
                                 ((x + 1, y), (x, y + 1))
                                 for x in range(SIZE_Y - 1)
                                 for y in range(SIZE_Y - 1)
                             ], weight=Diagonals_Weight)

        # remove obstacle nodes and edges
        for x in range(SIZE_X):
            for y in range(SIZE_Y):
                if DSM[x][y] == 1.:
                    G.remove_node((x, y))

        self.all_pairs_distances = nx.floyd_warshall(G)
        self.all_pairs_shortest_path = dict(nx.all_pairs_dijkstra_path(G))


        if PRINT_FLAG:
            path = self.all_pairs_shortest_path[(3,10)][(10,3)]
            path_edges = set(zip(path, path[1:]))
            nx.draw_networkx(G, pos=pos, labels=labels, font_size=5, with_labels=True, node_size=50)

            nx.draw_networkx_nodes(G, pos, nodelist=path, node_color='g')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[0]], node_color='b')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[1]], node_color='black')
            nx.draw_networkx_nodes(G, pos, nodelist=[path[-1]], node_color='r')
            # nx.draw_networkx_edges(G, pos, edgelist=path_edges, edge_color='r', width=10)
            plt.axis('equal')
            plt.show()

        return G

    def set_initial_state(self, state: State, episode_number, input_epsilon=None):
        pass


    def update_context(self, new_state: State, reward, is_terminal, EVALUATE=True):
        pass


    def get_action(self, state: State, evaluate=False)-> AgentAction:
        my_pos = state.my_pos.get_tuple()
        enemy_pos = state.enemy_pos.get_tuple()

        # all potential targets
        points_in_enemy_los = DICT_POS_LOS[enemy_pos]

        # find closest point in enemy line of sight
        best_distance = np.inf
        closest_target = None
        for point in points_in_enemy_los:
            dist_me_point = self.all_pairs_distances[my_pos][point]
            if dist_me_point < best_distance:
                best_distance = dist_me_point
                closest_target = point

        # get first step in path to closest_target
        path_to_closest_target = self.all_pairs_shortest_path[my_pos][closest_target]
        first_step = path_to_closest_target[1]
        delta_x =  first_step[0]-my_pos[0]
        delta_y = first_step[1] - my_pos[1]

        a=-1
        if NUMBER_OF_ACTIONS==9:
            a = self.get_action_9_actions(delta_x, delta_y)
        else:
            a = self.get_action_4_actions(delta_x, delta_y)

        if PRINT_FLAG:
            # print graph for debug
            pos = dict((n, n) for n in self.G.nodes())  # Dictionary of all positions
            labels = dict(((i, j), (i, j)) for i, j in self.G.nodes())
            path_to_closest_enemy_los = path_to_closest_target
            points_in_enemy_los_edges = set(zip(points_in_enemy_los, points_in_enemy_los[1:]))
            nx.draw_networkx(self.G, pos=pos, labels=labels, font_size=5, with_labels=True, node_size=50)
            nx.draw_networkx_nodes(self.G, pos, nodelist=points_in_enemy_los, node_color='r')
            nx.draw_networkx_nodes(self.G, pos, nodelist=path_to_closest_enemy_los, node_color='g')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[my_pos], node_color='b')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[first_step], node_color='black')
            nx.draw_networkx_nodes(self.G, pos, nodelist=[closest_target], node_color='y')
            # nx.draw_networkx_edges(self.G, pos, edgelist=points_in_enemy_los_edges, edge_color='r', width=10)
            plt.axis('equal')
            plt.show()

        self._action = a

        return self._action

    def get_action_9_actions(self, delta_x, delta_y):
        """9 possible moves!"""
        if delta_x == 1 and delta_y == -1:
            a = AgentAction.TopRight
        elif delta_x == 1 and delta_y == 0:
            a = AgentAction.Right
        elif delta_x == 1 and delta_y == 1:
            a = AgentAction.BottomRight
        elif delta_x == 0 and delta_y == -1:
            a = AgentAction.Bottom
        elif delta_x == 0 and delta_y == 0:
            a = AgentAction.Stay
        elif delta_x == 0 and delta_y == 1:
            a = AgentAction.Top
        elif delta_x == -1 and delta_y == -1:
            a = AgentAction.BottomLeft
        elif delta_x == -1 and delta_y == 0:
            a = AgentAction.Left
        elif delta_x == -1 and delta_y == 1:
            a = AgentAction.TopLeft

        return a

    def get_action_4_actions(self, delta_x, delta_y):
        """4 possible moves!"""
        if delta_x == 1 and delta_y == 0:
            a = AgentAction.Right
        elif delta_x == 0 and delta_y == -1:
            a = AgentAction.Bottom
        elif delta_x == 0 and delta_y == 1:
            a = AgentAction.Top
        elif delta_x == -1 and delta_y == 0:
            a = AgentAction.Left

    def type(self) -> AgentType:
        return self._type

    def get_epsolon(self):
        return self._epsilon

    def save_model(self, episodes_rewards, save_folder_path, color):
        pass


if __name__ == '__main__':
    PRINT_FLAG = True
    GP = Greedy_player()

    blue_pos = Position(3, 10)
    red_pos = Position(10, 3)
    ret_val = State(my_pos=blue_pos, enemy_pos=red_pos)

    a = GP.get_action(ret_val)
    print("The action to take is: ", a)
