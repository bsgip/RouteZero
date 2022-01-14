import osmnx as ox
import geopandas as gpd
import pandas as pd
import srtm
import numpy as np
import matplotlib.pyplot as plt
from shapely import wkt
import json


class RoadGraph:
    def __init__(self, map_name=None, map_center=None, sqr_dist=10, presaved=False):
        self.map_name = map_name
        self.map_center = map_center
        self.sqr_distance = sqr_dist
        self.elevation_data = srtm.get_data()
        if presaved:
            self.df_nodes = pd.read_csv('./map_data/'+self.map_name+'_nodes.csv', index_col='osmid')
            self.df_edges = pd.read_csv('./map_data/'+self.map_name+'_edges.csv', index_col=['u','v'])
            self.graph = ox.io.load_graphml(filepath='./map_data/'+self.map_name+'.graphml')
            with open('map_data/' + self.map_name + '.json', 'r') as f:
                map_info = json.load(f)
                self.map_name = map_info['map_name']
                self.map_center = map_info['map_center']
                self.sqr_distance = map_info['sqr_dist']

        else:
            G = ox.graph_from_address('9 tarin st, mayfield east, nsw', network_type='drive', dist=1000 * sqr_dist)
            G = ox.add_edge_speeds(G)  # impute speed on all edges missing data
            G = ox.add_edge_travel_times(G)  # calculate travel time (seconds) for all edges
            G = ox.add_edge_bearings(G)
            gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)
            df_nodes = pd.DataFrame(gdf_nodes)      # pandas data frames have less problems saving and loading data
            df_edges = pd.DataFrame(gdf_edges)
            self.df_nodes = df_nodes
            self.df_edges = df_edges
            self.graph = G

    def save(self):

            self.df_nodes.to_csv('./map_data/' + self.map_name+'_nodes.csv')
            self.df_edges.to_csv('./map_data/' + self.map_name + '_edges.csv')
            ox.io.save_graphml(self.graph, filepath='./map_data/'+self.map_name+'.graphml')
            map_info = {"map_name":self.map_name,
                        "map_center":self.map_center,
                        "sqr_dist":self.sqr_distance}

            with open('map_data/'+self.map_name+'.json', 'w') as f:
                json.dump(map_info, f)

    def create_one_route(self, start_coord, end_coord):
        """ compute shortest path for a given starting and ending point, returns goe dataframe route (df_nodes_route) and osmid of the route (route) """
        orig = ox.distance.nearest_nodes(self.graph, X=start_coord[1], Y=start_coord[0])
        dest = ox.distance.nearest_nodes(self.graph, X=end_coord[1], Y=end_coord[0])
        route = ox.shortest_path(self.graph, orig, dest, weight="length")  # todo: maybe change this to 'travel_time'
        if (route is None) or (len(route) == 1):    # no route between points
            return [], pd.DataFrame(), pd.DataFrame()

        df_nodes_route = self.df_nodes.loc[route]  # subset of df_nodes for the route (list of osmid) only

        for i in range(len(route) - 1):
            if i == 0:
                df_edges_route = self.df_edges.xs((route[i], route[i + 1]), level=('u', 'v'), drop_level=False)
            else:
                df_edges_route = df_edges_route.append(
                    self.df_edges.xs((route[i], route[i + 1]), level=('u', 'v'), drop_level=False).iloc[0])

        distance_travelled = np.hstack((0, np.cumsum(df_edges_route['length'].to_numpy())))
        node_elevations = []  # ??
        # to do, some kind of finer work if distnace between nodes is too far for elevation profiles to be good?
        for i, node in enumerate(df_nodes_route.iterrows()):
            node_elevations.append(self.elevation_data.get_elevation(node[1]['y'], node[1]['x']))

        edge_bearing = df_edges_route['bearing'].to_numpy()
        prev_edge_speeds = np.hstack((0, df_edges_route['speed_kph'].to_numpy()))
        node_bearing_change = (np.hstack((0, np.diff(edge_bearing), 0)) + 180) % (360) - 180
        # (phases + np.pi) % (2 * np.pi) - np.pi

        other_bearing_changes = [np.array((0))]
        con_highway_vals = [self.df_edges.xs(route[0], level='u')['highway'].values]
        for i in range(1, len(route)):
            connecting_edges = self.df_edges.xs(route[i], level='u')['bearing']
            bearing_changes = connecting_edges.to_numpy() - edge_bearing[i - 1]
            other_bearing_changes.append((bearing_changes + 180) % (360) - 180)
            con_highway_vals.append(self.df_edges.xs(route[i], level='u')['highway'].values)
        other_bearing_changes[-1] = np.array(0)

        # set highway aspect of first and last node to be a stop
        df_nodes_route.iloc[0, df_nodes_route.columns.get_loc('highway')] = 'stop'
        df_nodes_route.iloc[-1, df_nodes_route.columns.get_loc('highway')] = 'stop'

        df_nodes_route['distance_travelled'] = distance_travelled
        df_nodes_route['elevation'] = node_elevations
        df_nodes_route['bearing_change'] = node_bearing_change
        df_nodes_route['other_bearing_changes'] = other_bearing_changes
        df_nodes_route['con_highway_vals'] = con_highway_vals
        df_nodes_route['prev_edge_speed_kph'] = prev_edge_speeds

        return route, df_nodes_route, df_edges_route

    # def plot(self):       # todo: fix me
    #     ox.plot_graph_folium(self.graph, popup_attribute="name", weight=2, color="#8b0000")
    #     ox.plot.plot_graph(self.graph)
    #     plt.show()

    def plot_route(self, route):
        ox.plot.plot_graph_route(self.graph, route, route_color='r', route_linewidth=4, route_alpha=0.5, orig_dest_size=100,
                                 ax=None)
        plt.show()


if __name__=='__main__':
    map_name = 'test'
    map_center = '23 Maitland Road, Mayfield NSW'
    sqr_dist = 3        # kms

    # roadGraph = RoadGraph(map_name=map_name, map_center=map_center, sqr_dist=sqr_dist)

    ## testing saving and loading
    # roadGraph.save()
    roadGraph = RoadGraph(map_name='test',presaved=True)
    # roadGraph.plot()


    ##
    start_point = '23 Maitland Road, Mayfield NSW'
    end_point = '142 Doran St, Carrington, NSW'

    start_coords = ox.geocoder.geocode(start_point)
    end_coords = ox.geocoder.geocode(end_point)


    route, df_nodes_route, df_edges_route = roadGraph.create_one_route(start_coords, end_coords)
    roadGraph.plot_route(route)


