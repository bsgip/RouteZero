import osmnx as ox
import geopandas as gpd
import srtm
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import IFrame This only works in fucking jupyter
elevation_data = srtm.get_data()    # instanciate srtm.py

sqr_dist = 3        # kms
G = ox.graph_from_address('9 tarin st, mayfield east, nsw', network_type='drive',dist=1000*sqr_dist)
ox.io.save_graphml(G, filepath='map_1.graphml', gephi=False, encoding='utf-8')
# Option B : Load an already saved GPS datas
# G = ox.io.load_graphml('map_1.graphml', node_dtypes=None, edge_dtypes=None, graph_dtypes=None)

G = ox.add_edge_speeds(G)                                                                                               # impute speed on all edges missing data
G = ox.add_edge_travel_times(G)     # calculate travel time (seconds) for all edges
G = ox.add_edge_bearings(G)

gdf_nodes, gdf_edges = ox.graph_to_gdfs(G, nodes=True, edges=True)                                                      # Go with GeoPandas GeoDataFrames

# start_point = '9 Tarin St, Mayfield East, NSW'
start_point = '23 Maitland Road, Mayfield NSW'
end_point = '142 Doran St, Carrington, NSW'

start_coords = ox.geocoder.geocode(start_point)
end_coords = ox.geocoder.geocode(end_point)


def create_one_route(G, gdf_nodes, gdf_edges, start_point, end_point):
    """ compute shortest path for a given starting and ending point, returns goe dataframe route (gdf_nodes_route) and osmid of the route (route) """
    orig = ox.distance.nearest_nodes(G, X=start_point[1], Y=start_point[0])
    dest = ox.distance.nearest_nodes(G, X=end_point[1], Y=end_point[0])
    route = ox.shortest_path(G, orig, dest, weight="length")        # todo: maybe change this to 'travel_time'
    if (route is None) or (len(route) == 1):
        return [], gpd.GeoDataFrame()

    gdf_nodes_route = gdf_nodes.loc[route]  # subset of gdf_nodes for the route (list of osmid) only

    for i in range(len(route)-1):
        if i==0:
            gdf_edges_route = gdf_edges.xs((route[i], route[i+1]), level=('u', 'v'), drop_level=False)
        else:
            gdf_edges_route = gdf_edges_route.append(gdf_edges.xs((route[i], route[i+1]), level=('u', 'v'), drop_level=False))


    distance_travelled = np.hstack((0, np.cumsum(gdf_edges_route['length'].to_numpy())))
    node_elevations = []  # ??
    for i, node in enumerate(gdf_nodes_route.iterrows()):
        node_elevations.append(elevation_data.get_elevation(node[1]['y'], node[1]['x']))

    edge_bearing = gdf_edges_route['bearing'].to_numpy()
    prev_edge_speeds = np.hstack((0,gdf_edges_route['speed_kph'].to_numpy()))
    node_bearing_change = np.hstack((0,np.diff(edge_bearing),0))

    other_bearing_changes = [np.array((0))]
    con_highway_vals = [gdf_edges.xs(route[0],level='u')['highway'].values]
    for i in range(1,len(route)):
        connecting_edges = gdf_edges.xs(route[i],level='u')['bearing']
        bearing_changes = connecting_edges.to_numpy() - edge_bearing[i-1]
        other_bearing_changes.append(bearing_changes )
        con_highway_vals.append(gdf_edges.xs(route[i],level='u')['highway'].values)
    other_bearing_changes[-1] = np.array(0)

    # set highway aspect of first and last node to be a stop
    gdf_nodes_route.iloc[0,gdf_nodes_route.columns.get_loc('highway')] = 'stop'
    gdf_nodes_route.iloc[-1, gdf_nodes_route.columns.get_loc('highway')] = 'stop'

    gdf_nodes_route['distance_travelled'] = distance_travelled
    gdf_nodes_route['elevation'] = node_elevations
    gdf_nodes_route['bearing_change'] = node_bearing_change
    gdf_nodes_route['other_bearing_changes'] = other_bearing_changes
    gdf_nodes_route['con_highway_vals'] = con_highway_vals
    gdf_nodes_route['prev_edge_speed_kph'] = prev_edge_speeds

    return route, gdf_nodes_route, gdf_edges_route

route, gdf_nodes_route, gdf_edges_route = create_one_route(G, gdf_nodes, gdf_edges, start_coords, end_coords)


# plot the street network with folium
m1 = ox.plot_graph_folium(G, popup_attribute="name", weight=2, color="#8b0000")

# save as html file then display map as an iframe
filepath = "graph.html"
m1.save(filepath)

ox.plot.plot_graph_route(G, route, route_color='r', route_linewidth=4, route_alpha=0.5, orig_dest_size=100, ax=None)
plt.show()


gdf_nodes_route.plot(kind='line', x='distance_travelled', y="elevation", marker='.', figsize=(20,4))
plt.show()

gdf_nodes_route.plot(kind='line', x='distance_travelled', y='prev_edge_speed_kph')
plt.show()


#### Generate speed profile from route node and edge data

# TODO: an option to do freeflow node speeds would be simplest

# Now work out 'realistic speeds at nodes'
# TODO: could also factor in change in street type when generating
# TODO: use the roundabout poperty of junction from edges
# could use the highway attributes to work out speed adjustments?? https://wiki.openstreetmap.org/wiki/Key:highway

# TODO: synthetic speed profiles form other source??
def assign_psudo_realistic_node_speeds(gdf_nodes_route):
    node_speeds = []
    for i in range(len(gdf_nodes_route)):
        node = gdf_nodes_route.iloc[i]
        node_highway_val = node['highway']
        if node_highway_val == 'stop' or node_highway_val == 'traffic_signals' or node_highway_val == 'give_way' or node_highway_val =='crossing':
            node_speeds.append(0.)
        else:
            # is turnoff?
            is_intersection = node['street_count'] > 2
            turnoffon = not node['bearing_change'] == np.min(node['other_bearing_changes'])
            is_left = node['bearing_change'] < 0
            if abs(node['bearing_change']) > 70:
                corner = int(3)
                corner_mod_speed = node['prev_edge_speed_kph'] * 0.4
            elif abs(node['bearing_change']) > 45:
                corner = int(2)
                corner_mod_speed= node['prev_edge_speed_kph'] * 0.6
            elif abs(node['bearing_change']) < 22.5:
                corner = int(1)
                corner_mod_speed = node['prev_edge_speed_kph'] * 0.8
            else:
                corner = int(0)
                corner_mod_speed = node['prev_edge_speed_kph'] * 1.0 # free flow speed

            # if not intersection
            if not is_intersection:
                node_speeds.append(corner_mod_speed)

            else: # if it is an intersection
                if not turnoffon:
                    # TODO: check if crossing a more major road
                    node_speeds.append(corner_mod_speed)
                else:   # if turning on or off rather than going straight
                    # TODO: check if crossing a more major road
                    if is_left:
                        node_speeds.append(corner_mod_speed)
                    else:
                        node_speeds.append(0.)
    gdf_nodes_route['speed_kph'] = node_speeds
    return gdf_nodes_route

gdf_nodes_route = assign_psudo_realistic_node_speeds(gdf_nodes_route)

## generate trapezoidal speed profile
max_accel = 
