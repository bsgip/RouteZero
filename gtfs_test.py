import gtfs_functions as gtfs
import pandas as pd
pd.options.display.max_columns = None
import numpy as np
import webbrowser
from shapely.geometry import LineString
import geopandas as gpd
import folium
import branca
import matplotlib.pyplot as plt
# todo: for final processing don't just do for busiest day
routes, stops, stop_times, trips, shapes = gtfs.import_gtfs("./data/full_greater_sydney_gtfs_static.zip", busiest_date=True)

'''
                Filter to only bus routes
'''
## see if we can get only bus routes
# bus services have route types in teh 700 so >=700 and < 800
total_routes = len(routes)
total_trips = len(trips)
bus_routes = routes.loc[(routes.route_type>=700) & (routes.route_type<800)].reset_index()


'''
                CUT DOWN TO A SUBSET OF ROUTES and the data that corresponds to it
                Below cuts by the sydney buses network
                for later use it might be good to cut by a set of route names
'''
#
# cut down to sydney bus network routes
sydney_bus_routes = routes.loc[routes.route_desc=='Sydney Buses Network']
sydney_bus_route_ids = sydney_bus_routes['route_id'].unique()

# we can cut down the trips based on the route ids
sydney_bus_trips = trips.loc[trips.route_id.isin(sydney_bus_route_ids)]     # this has shape id

# can cut down stop_times based on trip id
sydney_bus_stop_times = stop_times.loc[stop_times.trip_id.isin(sydney_bus_trips['trip_id'])]

# can cut down stops based on stop id from stop times
sydney_bus_stops = stops.loc[stops.stop_id.isin(sydney_bus_stop_times['stop_id'])]

# cut down shapes based on shape id from above
sydney_bus_shapes = shapes.loc[shapes.shape_id.isin(sydney_bus_trips['shape_id'])]


## get frequency of stop and lines for specific time windows (given by cuttoffs) frequency is how many minutes between a bus
cutoffs = [0,6,9,15,19,22,24]       # times of day into which to aggregate the data (i.e. bin edges
stop_freq = gtfs.stops_freq(sydney_bus_stop_times, sydney_bus_stops, cutoffs=cutoffs)
line_freq = gtfs.lines_freq(sydney_bus_stop_times, sydney_bus_trips, sydney_bus_shapes, sydney_bus_routes, cutoffs=cutoffs)



""" 
                Plotting some things using gtfs_functions

"""
def folium_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)

condition_dir = stop_freq.dir_id == 'Inbound'
condition_window = stop_freq.window == '6:00-9:00'

gdf = stop_freq.loc[(condition_dir & condition_window),:].reset_index()

m = gtfs.map_gdf(gdf = gdf,
              variable = 'ntrips',
              colors = ["#d13870", "#e895b3" ,'#55d992', '#3ab071', '#0e8955','#066a40'],
              tooltip_var = ['frequency'] ,
              tooltip_labels = ['Frequency: '],
              breaks = [10, 20, 30, 40, 120, 200])

folium_open(m, 'test.html')


# # Line frequencies
# condition_dir = line_freq.dir_id == 'Inbound'
# condition_window = line_freq.window == '6:00-9:00'
#
# gdf = line_freq.loc[(condition_dir & condition_window),:].reset_index()
#
# m = gtfs.map_gdf(gdf = gdf,
#               variable = 'ntrips',
#               colors = ["#d13870", "#e895b3" ,'#55d992', '#3ab071', '#0e8955','#066a40'],
#               tooltip_var = ['route_name'] ,
#               tooltip_labels = ['Route: '],
#               breaks = [5, 10, 20, 50])
# #
# folium_open(m, 'test.html')

'''
                What do we really want? 
                Per route, the speed the bus does between each of its stops
                how to work this out? 
                pick a route, get all the trips for that route
                get the stop times for those trips
'''

# filter down to a subset of routes based on route short name
# route_short_names = ["305", "320"]
route_short_names = [a[0].replace("s","").replace("w","") for a in pd.read_csv('./data/zenobe_routes.csv').values]

subset_routes = sydney_bus_routes.loc[sydney_bus_routes['route_short_name'].isin(route_short_names)]
subset_trips = sydney_bus_trips.loc[sydney_bus_trips.route_id.isin(subset_routes['route_id'])]
subset_stop_times = stop_times.loc[stop_times.trip_id.isin(subset_trips['trip_id'])]
subset_stops = stops.loc[stops.stop_id.isin(subset_stop_times['stop_id'])]
subset_shapes = shapes.loc[shapes.shape_id.isin(subset_trips['shape_id'])]

# below considers more of the bus networks

# subset_routes = routes.loc[routes['route_short_name'].isin(route_short_names)]
# subset_trips = trips.loc[trips.route_id.isin(subset_routes['route_id'])]
# subset_stop_times = stop_times.loc[stop_times.trip_id.isin(subset_trips['trip_id'])]
# subset_stops = stops.loc[stops.stop_id.isin(subset_stop_times['stop_id'])]
# subset_shapes = shapes.loc[shapes.shape_id.isin(subset_trips['shape_id'])]

# sort routes by trip and stop sequence
subset_stop_times.sort_values(by=['trip_id','stop_sequence'], ascending=True, inplace=True)

# For each trip, break each segment (pair of consecutive stops) into start and end points
seg_start = subset_stop_times.drop(subset_stop_times.groupby(by='trip_id').tail(1).index, axis=0).reset_index()
seg_start = seg_start[['trip_id','departure_time','stop_id','route_id','direction_id','geometry','stop_name','shape_id','shape_dist_traveled']]
seg_start.columns = ['trip_id','start_time','start_stop_id','route_id','direction_id','start_geometry','start_stop_name','shape_id','start_dist_traveled']
seg_end = subset_stop_times.drop(subset_stop_times.groupby(by='trip_id').head(1).index, axis=0).reset_index()
seg_end = seg_end[['arrival_time','stop_id','geometry','stop_name','shape_dist_traveled']]
seg_end.columns = ['end_time','end_stop_id','end_geometry','end_stop_name','end_dist_traveled']

# merge start and end stop information into segment information
segment_df = pd.concat([seg_start,seg_end],axis=1).drop_duplicates()
segment_df['distance'] = segment_df['end_dist_traveled'] - segment_df['start_dist_traveled']
segment_df['duration'] = segment_df['end_time'] - segment_df['start_time']
segment_df['speed_mps'] = (segment_df['distance'] / segment_df['duration']).replace([np.inf, -np.inf], np.nan)
segment_df['speed_kph'] = segment_df['speed_mps'] * 3.6
segment_df['segment_id'] = segment_df['start_stop_id'] + "-" + segment_df['end_stop_id'] # create a segment name
# merge geometry points into linestring for each seg
segment_df['geometry'] = [LineString([p1, p2]) for p1, p2 in zip(segment_df['start_geometry'].values,segment_df['end_geometry'].values)]
# drop unneeded columns
segment_df.drop(columns=['start_geometry','end_geometry'],inplace=True)

'''
Get average speed of each segment per route and also per time window???
'''

cutoffs = [0, 6, 9, 15, 19, 22, 24]
labels = [str(cutoffs[i]) + '-' + str(cutoffs[i+1]) for i in range(0, len(cutoffs)-1)]
segment_df['window'] = pd.cut(segment_df['start_time']/3600,bins=cutoffs,right=False, labels=labels)

# create a new data frame containing speeds averaged by segment and route (sorting by segment does by direction inherently)
speeds = segment_df.groupby(by=['segment_id','window','route_id','direction_id'])['speed_mps'].mean().reset_index(name="av_speed_mps")
speeds['count'] = segment_df.groupby(by=['segment_id','window','route_id','direction_id'])['speed_mps'].count().values
speeds.dropna(inplace=True)


# average speeds over routes broken into windows
route_speeds = speeds.groupby(by=['route_id','direction_id','window'])['av_speed_mps'].mean().reset_index(name='av_speed_mps')
route_speeds.dropna(inplace=True)

gdf = gpd.GeoDataFrame(subset_shapes)
gdf['route_id'] = "nan"
gdf['av_speed_mps'] = np.nan
for i, r_id in enumerate(route_speeds.route_id.unique()):
    shape_id = subset_trips.loc[(subset_trips.route_id==r_id) & (subset_trips.direction_id==1)]['shape_id'].values
    if len(shape_id):
        index = gdf[gdf.shape_id==shape_id[0]].index[0]
        gdf.at[index,'route_id']= r_id
        r_speed =route_speeds.loc[(route_speeds.route_id==r_id) &(route_speeds.direction_id==1) & (route_speeds.window=='9-15')].av_speed_mps.values
        if len(r_speed):
            gdf.at[index, 'av_speed_mps'] = r_speed[0]
gdf.dropna(inplace=True)


minx, miny, maxx, maxy = gdf.geometry.total_bounds

centroid_lat = miny + (maxy - miny) / 2
centroid_lon = minx + (maxx - minx) / 2

m = folium.Map(location=[centroid_lat, centroid_lon],
               tiles='cartodbpositron', zoom_start=12)
gdf.crs = {'init':'epsg:4326'}

colorscale = branca.colormap.linear.YlGnBu_09.scale(0, 30)
def style_function(feature):
    return {
        'fillOpacity': 0.5,
        'weight': 3,  # math.log2(feature['properties']['speed'])*2,
        'color':colorscale(feature['properties']['av_speed_mps'])
    }

# my code for lines
geo_data = gdf.__geo_interface__
folium.GeoJson(
    geo_data,
    style_function=style_function,
    tooltip=folium.features.GeoJsonTooltip(fields=['route_id','av_speed_mps'],
                                           # aliases=tooltip_labels,
                                           labels=True,
                                           sticky=False)
).add_to(m)


folium_open(m, 'test.html')


# average bus speed by window
# bus_speeds_all = speeds.groupby(by=['window'])['av_speed_mps'].mean().reset_index(name='av_speed_mps')
# bus_speeds_all.dropna(inplace=True)
# bus_speeds_all.sort_values(by='window',inplace=True)

#
# for route_id in route_speeds['route_id'].unique().tolist():
#     tmp = route_speeds.loc[route_speeds.route_id==route_id].sort_values(by='window', ascending=True)
#     av_speed = tmp['av_speed_mps']*3.6
#     windows = tmp['window']
#     # x_vals = [([0]+x[1]) for w in windows]
#     plt.plot(windows, av_speed)
#     plt.show()

# plt.plot(speeds[speeds.segment_id=='2000133-200029']['window'],speeds[speeds.segment_id=='2000133-200029']['av_speed_mps'])

# duration = subset_stop_times.groupby(by='trip_id').apply(lambda x: (x['arrival_time'].shift(-1)-x['departure_time'])[:-1]).values
# start_stop_id = subset_stop_times.groupby(by='trip_id')



'''
                GTFS_functions way of getting segments
                This is really slow and sometimes fails
                might only work for busses
'''
# segments_gdf = gtfs.cut_gtfs(sydney_bus_stop_times, sydney_bus_stops, sydney_bus_shapes)
# segments_gdf.head()
#
#
# seg_freq = gtfs.segments_freq(segments_gdf, sydney_bus_stop_times, sydney_bus_routes, cutoffs=cutoffs)
# speeds = gtfs.speeds_from_gtfs(sydney_bus_routes, sydney_bus_stop_times, segments_gdf, cutoffs = cutoffs)
#
#
# # plot speeds vs hour in a nice way
# import plotly.graph_objects as go
#
# example2 = speeds.loc[(speeds.s_st_name == 'Chetwynd Rd at Charlotte St') & (speeds.route_name == 'All lines')].sort_values(
#     by='stop_seq')
# example2['hour'] = example2.window.apply(lambda x: int(x.split(':')[0]))
# example2.sort_values(by='hour', ascending=True, inplace=True)
#
# fig = go.Figure()
#
# trace = go.Scatter(
#     name='Speed',
#     x=example2.hour,
#     y=example2.speed_kmh,
#     mode='lines',
#     line=dict(color='rgb(31, 119, 180)'),
#     fillcolor='#F0F0F0',
#     fill='tonexty',
#     opacity=0.5)
#
# data = [trace]
#
# layout = go.Layout(
#     yaxis=dict(title='Average Speed (km/h)'),
#     xaxis=dict(title='Hour of day'),
#     title='Average Speed by hour of day in stop Fillmore St & Bay St',
#     showlegend=False, template='simple_white')
#
# fig = go.Figure(data=data, layout=layout)
#
# # Get the labels in the X axis right
# axes_labels = []
# tickvals = example2.hour.unique()
#
# for i in range(0, len(tickvals)):
#     label = str(tickvals[i]) + ':00'
#     axes_labels.append(label)
#
# fig.update_xaxes(
#     ticktext=axes_labels,
#     tickvals=tickvals
# )
#
# # Add vertical lines
# y_max_value = example2.speed_kmh.max()
#
# for i in range(0, len(tickvals)):
#     fig.add_shape(
#         # Line Vertical
#         dict(
#             type="line",
#             x0=tickvals[i],
#             y0=0,
#             x1=tickvals[i],
#             y1=y_max_value,
#             line=dict(
#                 color="Grey",
#                 width=1
#             )
#         )
#     )
#
# # Labels in the edge values
# for i in range(0, len(tickvals)):
#     y_value = example2.loc[example2.hour == tickvals[i], 'speed_kmh'].values[0].round(2)
#     fig.add_annotation(
#         x=tickvals[i],
#         y=y_value,
#         text=str(y_value),
#     )
# fig.update_annotations(dict(
#     xref="x",
#     yref="y",
#     showarrow=True,
#     arrowhead=0,
#     ax=0,
#     ay=-18
# ))
#
# fig.update_yaxes(rangemode='tozero')
#
# fig.show()