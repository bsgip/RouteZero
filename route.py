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
        """ compute shortest path for a given starting and ending point, returns goe dataframe node_ids (df_nodes_route) and osmid of the node_ids (node_ids) """
        orig = ox.distance.nearest_nodes(self.graph, X=start_coord[1], Y=start_coord[0])
        dest = ox.distance.nearest_nodes(self.graph, X=end_coord[1], Y=end_coord[0])
        node_ids = ox.shortest_path(self.graph, orig, dest, weight="length")  # todo: maybe change this to 'travel_time'
        if (node_ids is None) or (len(node_ids) == 1):    # no node_ids between points
            return [], pd.DataFrame(), pd.DataFrame()

        df_nodes_route = self.df_nodes.loc[node_ids]  # subset of df_nodes for the node_ids (list of osmid) only

        for i in range(len(node_ids) - 1):
            if i == 0:
                df_edges_route = self.df_edges.xs((node_ids[i], node_ids[i + 1]), level=('u', 'v'), drop_level=False)
            else:
                df_edges_route = df_edges_route.append(
                    self.df_edges.xs((node_ids[i], node_ids[i + 1]), level=('u', 'v'), drop_level=False).iloc[0])

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
        con_highway_vals = [self.df_edges.xs(node_ids[0], level='u')['highway'].values]
        for i in range(1, len(node_ids)):
            connecting_edges = self.df_edges.xs(node_ids[i], level='u')['bearing']
            bearing_changes = connecting_edges.to_numpy() - edge_bearing[i - 1]
            other_bearing_changes.append((bearing_changes + 180) % (360) - 180)
            con_highway_vals.append(self.df_edges.xs(node_ids[i], level='u')['highway'].values)
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

        route = Route(node_ids, df_nodes_route, df_edges_route)
        return route

    # def plot(self):       # todo: fix me
    #     ox.plot_graph_folium(self.graph, popup_attribute="name", weight=2, color="#8b0000")
    #     ox.plot.plot_graph(self.graph)
    #     plt.show()

    def plot_route(self, node_ids):
        if len(node_ids) ==1:
            ox.plot.plot_graph_route(self.graph, node_ids[0], route_color='r', route_linewidth=4, route_alpha=0.5, orig_dest_size=100,
                                 ax=None)
        elif len(node_ids) > 1:
            ox.plot.plot_graph_route(self.graph, node_ids)       ## todo: finish this~!
        plt.show()


class Route():
    def __init__(self, node_ids, df_nodes_route, df_edges_route):
        self.node_ids = [node_ids]
        self.df_nodes_route = [df_nodes_route]
        self.df_edges_route = [df_edges_route]
        self.ref_distances = []
        self.ref_speeds = []
        self.speed_profiles = []

    def create_trapezoidal_speed_profile(self, max_accel=1.5, max_deccel=2.5, dist_res=5):
        max_deccel = np.abs(max_deccel)     # enforce this is a positve number
        for k, (ref_distances, ref_speeds) in enumerate(zip(self.ref_distances, self.ref_speeds)):
            total_distance = ref_distances[-1]
            n_segs = int(np.ceil(total_distance / dist_res))

            a1 = np.zeros(n_segs + 1, )
            v1 = np.zeros(n_segs + 1, )
            delta_t1 = np.zeros(n_segs + 1, )

            i = 0
            speed_ref = ref_speeds[0]
            dist_ref = ref_distances[0]
            last_speed_ref = 0.
            k = 0
            # forward pass for accelerating after reference speed changes
            for d in range(n_segs):
                if d * dist_res >= dist_ref:
                    k = k + 1
                    if v1[d] > speed_ref:  # we shouldn't have gotten faster than the speed ref
                        v1[d] = speed_ref
                    last_speed_ref = speed_ref
                    speed_ref = ref_speeds[k]
                    dist_ref = ref_distances[k]

                target_speed = np.maximum(speed_ref, last_speed_ref)
                if v1[d] < speed_ref or v1[d] < last_speed_ref:
                    a1[d] = max_accel
                else:
                    a1[d] = 0

                if a1[d] > 1e-4:
                    delta_t = (-v1[d] + np.sqrt(v1[d] ** 2 + 2 * a1[d] * dist_res)) / 2
                elif v1[d] > 1e-4:
                    delta_t = dist_res / v1[d]
                else:  # not moving and not accelerating (stopped for 30sec)
                    delta_t = 30
                delta_t1[d] = delta_t  # time taken to traverse current segment
                v1[d + 1] = np.minimum(v1[d] + a1[d] * delta_t, target_speed)  # speed at start of next segment

            # backwards pass to enforce decceleration before slower speed ref
            a2 = np.zeros(n_segs + 1, )
            v2 = np.zeros(n_segs + 1, )
            delta_t2 = np.zeros(n_segs + 1, )
            for d in reversed(range(1, n_segs + 1)):
                if d * dist_res <= dist_ref:
                    if v2[d] > speed_ref:
                        v2[d] = speed_ref
                    k = k - 1
                    last_speed_ref = speed_ref
                    speed_ref = ref_speeds[k]
                    dist_ref = ref_distances[k]
                target_speed = np.maximum(speed_ref, last_speed_ref)
                if v2[d] < speed_ref or v2[d] < last_speed_ref:
                    a2[d] = max_deccel
                else:
                    a2[d] = 0

                if a2[d] > 1e-4:
                    delta_t = (-v2[d] + np.sqrt(v2[d] ** 2 + 2 * a2[d] * dist_res)) / 2
                elif v2[d] > 1e-4:
                    delta_t = dist_res / v2[d]
                else:  # not moving and not accelerating (stopped for 30sec)
                    delta_t = 30
                delta_t2[d] = delta_t  # time taken to traverse current segment
                v2[d - 1] = np.minimum(target_speed, v2[d] + a2[d] * delta_t)
            a2 = - a2  # invert this since we want it all to be accelerations

            # combine forward and backward profiles
            tmp = np.vstack((v1, v2))
            inds = np.argmin(tmp, axis=0)
            v = tmp[inds, np.arange(len(v1))]
            delta_t = np.vstack((delta_t1, delta_t2))[inds, np.arange(len(v1))]
            t = np.cumsum(delta_t) - delta_t[0]
            a = np.vstack((a1, a2))[inds, np.arange(len(v1))]
            d = dist_res * np.arange(n_segs+1)
            data = np.vstack((t,a,v,d))
            speed_profile = pd.DataFrame(data=data.T, columns=['time','acceleration','velocity','distance'])

            self.speed_profiles.append(speed_profile)

    def create_reference_speeds(self, method='psuedo'):
        if method=='psuedo':
            self._assign_psudo_realistic_node_speeds_()
        elif method=='free_flow':
            self._assign_free_flow_node_speeds_()
        else:
            print('Invalid method. Options {psuedo (default), free_flow}')
            return

        ref_speeds_list = []
        ref_distances_list = []
        for i, (df_nodes_route, df_edges_route) in enumerate(zip(self.df_nodes_route, self.df_edges_route)):
            node_speeds = df_nodes_route['speed_kph'].to_numpy() / 3.6  # convert to mps
            node_distance = df_nodes_route['distance_travelled'].to_numpy()
            edge_speeds = df_edges_route['speed_kph'].to_numpy() / 3.6  # convert to mps
            edge_is_roundabout = df_edges_route['junction'].to_numpy().astype(str) == 'roundabout'
            if method=='psuedo':
                edge_speeds = edge_speeds * (1 - 0.5 * edge_is_roundabout)  # TODO: do something better with roundabout
            edge_lengths = df_edges_route['length'].to_numpy()

            ref_speeds = []
            ref_distances = []
            for i in range(len(node_speeds) - 1):
                # the node speed and distance
                ref_speeds.append(node_speeds[i])
                ref_distances.append(node_distance[i])
                # the edge speed and some distance along it
                ref_speeds.append(edge_speeds[i])
                ref_distances.append(node_distance[i] + edge_lengths[i] / 2)
            ref_speeds.append(node_speeds[i + 1])
            ref_distances.append(node_distance[i + 1])
            # TODO: Need to make sure we don't have multiple zero speeds in a row
            ref_speeds_list.append(ref_speeds)
            ref_distances_list.append(ref_distances)
        self.ref_speeds = ref_speeds_list
        self.ref_distances = ref_distances_list

    def _assign_free_flow_node_speeds_(self):
        for k, df_nodes_route in enumerate(self.df_nodes_route):
            node_speeds = [0]
            for i in range(1, len(df_nodes_route)-1):
                node = df_nodes_route.iloc[i]
                node_speeds.append(node['prev_edge_speed_kph'])
            node_speeds.append(0)
            self.df_nodes_route[k]['speed_kph'] = node_speeds


    def _assign_psudo_realistic_node_speeds_(self):
        for k, df_nodes_route in enumerate(self.df_nodes_route):
            node_speeds = []
            for i in range(len(df_nodes_route)):
                node = df_nodes_route.iloc[i]
                node_highway_val = node['highway']
                if node_highway_val == 'stop' or node_highway_val == 'traffic_signals' or node_highway_val == 'give_way' or node_highway_val == 'crossing':
                    node_speeds.append(0.)
                else:
                    # is turnoff?
                    is_intersection = node['street_count'] > 2
                    turnoffon = not node['bearing_change'] == np.min(node['other_bearing_changes'])
                    is_left = node['bearing_change'] < 0
                    if abs(node['bearing_change']) > 70:
                        corner_mod_speed = node['prev_edge_speed_kph'] * 0.4
                    elif abs(node['bearing_change']) > 45:
                        corner_mod_speed = node['prev_edge_speed_kph'] * 0.6
                    elif abs(node['bearing_change']) > 22.5:
                        corner_mod_speed = node['prev_edge_speed_kph'] * 0.8
                    else:
                        corner_mod_speed = node['prev_edge_speed_kph'] * 1.0  # free flow speed

                    # if not intersection
                    if not is_intersection:
                        node_speeds.append(corner_mod_speed)

                    else:  # if it is an intersection
                        if not turnoffon:
                            # TODO: check if crossing a more major road
                            node_speeds.append(corner_mod_speed)
                        else:  # if turning on or off rather than going straight
                            # TODO: check if crossing a more major road
                            if is_left:
                                node_speeds.append(corner_mod_speed)
                            else:
                                node_speeds.append(0.)
            self.df_nodes_route[k]['speed_kph'] = node_speeds




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


    route = roadGraph.create_one_route(start_coords, end_coords)

    # ox.plot.plot_graph_routes(roadGraph.graph, route.node_ids)
    # plt.show()

    roadGraph.plot_route(route.node_ids)

    route.create_reference_speeds()

    route.create_trapezoidal_speed_profile()

    plt.subplot(3, 1, 1)
    plt.plot(route.speed_profiles[0]['time'],route.speed_profiles[0]['acceleration'])
    plt.xlabel('time (s)')
    plt.ylabel('acceleration (m/s^2)')

    plt.subplot(3, 1, 2)
    plt.plot(route.speed_profiles[0]['time'],route.speed_profiles[0]['velocity'])
    plt.xlabel('time (s)')
    plt.ylabel('velocity (m/s)')

    plt.subplot(3, 1, 3)
    plt.plot(route.speed_profiles[0]['time'],route.speed_profiles[0]['distance'])
    plt.xlabel('time (s)')
    plt.ylabel('distance (m)')
    plt.show()
