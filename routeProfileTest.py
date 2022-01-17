import osmnx as ox
import geopandas as gpd
import pandas as pd
import srtm
import numpy as np
import matplotlib.pyplot as plt
# from IPython.display import IFrame This only works in fucking jupyter
# elevation_data = srtm.get_data()    # instanciate srtm.py

from route import RoadGraph

map_name = 'test'
map_center = '23 Maitland Road, Mayfield NSW'
sqr_dist = 3  # kms

roadGraph = RoadGraph(map_name=map_name, map_center=map_center, sqr_dist=sqr_dist)

## testing saving and loading
# roadGraph.save()
# roadGraph = RoadGraph(map_name='test', presaved=True)
# roadGraph.plot()

##
start_point = '23 Maitland Road, Mayfield NSW'
end_point = '142 Doran St, Carrington, NSW'

start_coords = ox.geocoder.geocode(start_point)
end_coords = ox.geocoder.geocode(end_point)

route, df_nodes_route, df_edges_route = roadGraph.create_segment(start_coords, end_coords)
roadGraph.plot_route(route)


#### Generate speed profile from route node and edge data

# TODO: an option to do freeflow node speeds would be simplest

# Now work out 'realistic speeds at nodes'
# TODO: could also factor in change in street type when generating
# TODO: use the roundabout poperty of junction from edges
# could use the highway attributes to work out speed adjustments?? https://wiki.openstreetmap.org/wiki/Key:highway

# TODO: synthetic speed profiles form other source??
def assign_psudo_realistic_node_speeds(df_nodes_route):
    node_speeds = []
    for i in range(len(df_nodes_route)):
        node = df_nodes_route.iloc[i]
        node_highway_val = node['highway']
        if node_highway_val == 'stop' or node_highway_val == 'traffic_signals' or node_highway_val == 'give_way' or node_highway_val =='crossing':
            node_speeds.append(0.)
        else:
            # is turnoff?
            is_intersection = node['street_count'] > 2
            turnoffon = not node['bearing_change'] == np.min(node['other_bearing_changes'])
            is_left = node['bearing_change'] < 0
            if abs(node['bearing_change']) > 70:
                corner_mod_speed = node['prev_edge_speed_kph'] * 0.4
            elif abs(node['bearing_change']) > 45:
                corner_mod_speed= node['prev_edge_speed_kph'] * 0.6
            elif abs(node['bearing_change']) > 22.5:
                corner_mod_speed = node['prev_edge_speed_kph'] * 0.8
            else:
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
    df_nodes_route['speed_kph'] = node_speeds
    return df_nodes_route

df_nodes_route = assign_psudo_realistic_node_speeds(df_nodes_route)

## create reference speeds and distances from node and edge data

node_speeds = df_nodes_route['speed_kph'].to_numpy()/3.6       # convert to mps
node_distance = df_nodes_route['distance_travelled'].to_numpy()
edge_speeds = df_edges_route['speed_kph'].to_numpy()/3.6       # convert to mps
edge_is_roundabout = df_edges_route['junction'].to_numpy().astype(str) == 'roundabout'
edge_speeds = edge_speeds * (1 - 0.5*edge_is_roundabout)    # TODO: do something better with roundabout
edge_lengths = df_edges_route['length'].to_numpy()

ref_speeds = []
ref_distances = []
for i in range(len(node_speeds)-1):
    # the node speed and distance
    ref_speeds.append(node_speeds[i])
    ref_distances.append(node_distance[i])
    # the edge speed and some distance along it
    ref_speeds.append(edge_speeds[i])
    ref_distances.append(node_distance[i]+edge_lengths[i]/2)
ref_speeds.append(node_speeds[i+1])
ref_distances.append(node_distance[i+1])
# TODO: Need to make sure we don't have multiple zero speeds in a row

## generate trapezoidal speed profile from the reference speed and distances
## for the speed profiles we will work on the idea of zero order holded accelerations
# todo: Note this would be different from if we had measured data and we want to be at the speeds by the measurement points
max_accel = 1.5     # m/s^2
max_decel = 2.5     # m/s^2     # todo: enforce specified as positive value
time_res = 1        # s
dist_res = 5       # m
total_distance = ref_distances[-1]
n_segs = int(np.ceil(total_distance/dist_res))

a1 = np.zeros(n_segs+1,)
v1 = np.zeros(n_segs+1,)
delta_t1 = np.zeros(n_segs+1,)

i = 0
speed_ref = ref_speeds[0]
dist_ref = ref_distances[0]
last_speed_ref = 0.
k = 0
# forward pass for accelerating after reference speed changes
for d in range(n_segs):
    if d * dist_res >= dist_ref:
        k = k + 1
        if v1[d] > speed_ref:        # we shouldn't have gotten faster than the speed ref
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
        delta_t = (-v1[d] + np.sqrt(v1[d]**2 + 2*a1[d]*dist_res))/2
    elif v1[d] > 1e-4:
        delta_t = dist_res / v1[d]
    else:   # not moving and not accelerating (stopped for 30sec)
        delta_t = 30
    delta_t1[d] = delta_t       # time taken to traverse current segment
    v1[d+1] = np.minimum(v1[d] + a1[d] * delta_t, target_speed) # speed at start of next segment

# backwards pass to enforce decceleration before slower speed ref
a2 = np.zeros(n_segs+1,)
v2 = np.zeros(n_segs+1,)
delta_t2 = np.zeros(n_segs+1,)
for d in reversed(range(1,n_segs+1)):
    if d * dist_res <= dist_ref:
        if v2[d] > speed_ref:
            v2[d] = speed_ref
        k = k - 1
        last_speed_ref = speed_ref
        speed_ref = ref_speeds[k]
        dist_ref = ref_distances[k]
    target_speed = np.maximum(speed_ref, last_speed_ref)
    if v2[d] < speed_ref or v2[d] < last_speed_ref:
        a2[d] = max_decel
    else:
        a2[d] = 0

    if a2[d] > 1e-4:
        delta_t = (-v2[d] + np.sqrt(v2[d]**2 + 2*a2[d]*dist_res))/2
    elif v2[d] > 1e-4:
        delta_t = dist_res / v2[d]
    else:  # not moving and not accelerating (stopped for 30sec)
        delta_t = 30
    delta_t2[d] = delta_t  # time taken to traverse current segment
    v2[d-1] = np.minimum(target_speed,v2[d] + a2[d] * delta_t)
a2 = - a2   # invert this since we want it all to be accelerations

plt.plot(dist_res * np.arange(n_segs+1), v1)
plt.plot(dist_res * np.arange(n_segs+1), v2)
plt.scatter(ref_distances,ref_speeds)
plt.xlim([0,1000])
plt.show()

# combine forward and backward profiles
tmp = np.vstack((v1,v2))
inds = np.argmin(tmp, axis=0)
# v = np.minimum(v1, v2)
v = tmp[inds,np.arange(len(v1))]
delta_t = np.vstack((delta_t1,delta_t2))[inds,np.arange(len(v1))]
t = np.cumsum(delta_t) - delta_t[0]
a = np.vstack((a1,a2))[inds,np.arange(len(v1))]

plt.plot(dist_res * np.arange(n_segs+1), v1*3.6)
plt.plot(dist_res * np.arange(n_segs+1), v2*3.6)
plt.scatter(ref_distances,np.array(ref_speeds)*3.6)
plt.plot(dist_res * np.arange(n_segs+1), v*3.6, color='k', linewidth=2)
plt.xlim([0,1000])
plt.xlabel('distance travelled (m)')
plt.ylabel('speed km/h')
plt.show()

# plt.plot(dist_res * np.arange(n_segs+1), a)
# plt.show()
