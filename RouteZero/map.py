import geopandas as gpd
import numpy as np
import pandas as pd
import branca
import folium
import webbrowser

"""

    Functions for plotting results on a map
    
"""

def _folium_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)

def route_energy_map(trip_data, energy_consumption, shapes, window=None, mode='max'):

    tmp = trip_data.copy()
    tmp['EC'] = energy_consumption

    # if window is not None:
    #     tmp = tmp[tmp['trip_start_time'] > ]

    ## Prepare data for plotting on a map
    shape_ids = trip_data.shape_id

    gdf = gpd.GeoDataFrame(shapes)
    gdf['route_id'] = "nan"
    gdf['max_EC_total'] = np.nan
    gdf['max_EC_km'] = np.nan

def _create_gdf_of_value(trips_data, shapes, value, window=None, mode='max'):
    """
    creates a geopandas dataframe that has route shapes and an associated value so that routes can be plotted on a map
    and color coded by value
    :param trips_data: summarised trips dataframe
    :param shapes: shapes dataframe
    :param value: the value we wish to show in our plot, needs to be a list of the same length as trips_data, one value for each row
    :param window: optional list with two hour values specifying to only look at trips that start in that hour window
    :param mode: when aggregating multiple results from within a window, what mode to use 'max', 'mean', or 'min'
    :return:
    """

    tmp = trips_data.copy()
    tmp['start_hour'] = np.mod(tmp['trip_start_time']/3600,24)
    tmp['val'] = value

    # filter to specified window
    if window is not None:
        tmp = tmp[(tmp['start_hour'] > window[0]) & ((tmp['start_hour'] < window[1]) )]

    # apply mode of aggregation
    if mode=='max':
        shape_val = tmp.groupby(by=['shape_id','route_short_name','route_id'])['val'].max().reset_index()
    elif mode=='mean':
        shape_val = tmp.groupby(by=['shape_id','route_short_name','route_id'])['val'].mean().reset_index()
    elif mode=='min':
        shape_val = tmp.groupby(by=['shape_id','route_short_name','route_id'])['val'].min().reset_index()
    else:
        print('invalid mode')

    ## Prepare data for plotting on a map
    gdf = gpd.GeoDataFrame(shapes[shapes['shape_id'].isin(tmp['shape_id'].unique().tolist())])
    gdf['route_id'] = "nan"
    gdf['route_short_name'] = "nan"
    gdf['val'] = np.nan

    for i, s in gdf.iterrows():
        shape_id = s['shape_id']
        val = shape_val[shape_val.shape_id==shape_id].val
        route_id = shape_val[shape_val.shape_id==shape_id].route_id.to_numpy()[0]
        route_short_name = shape_val[shape_val.shape_id == shape_id].route_short_name.to_numpy()[0]

        # if i==1090:
        #     print('here')
        gdf.at[i, 'val'] = np.max(val.values)
        gdf.at[i, 'route_id'] = route_id
        gdf.at[i, 'route_short_name'] = route_short_name
        # print(i)
    gdf.dropna(inplace=True)

    return gdf

def _create_gdf_map(gdf, map_title, colorbar_str):
    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    centroid_lat = miny + (maxy - miny) / 2
    centroid_lon = minx + (maxx - minx) / 2

    ## create a map of total energy consumption
    m = folium.Map(location=[centroid_lat, centroid_lon],
                   tiles='cartodbpositron', zoom_start=12)
    gdf.crs = {'init': 'epsg:4326'}

    colorscale = branca.colormap.linear.YlGnBu_09.scale(gdf['val'].min(), gdf['val'].max())

    def style_function(feature):
        return {
            'fillOpacity': 0.5,
            'weight': 3,  # math.log2(feature['properties']['speed'])*2,
            'color': colorscale(feature['properties']['val'])
        }

    # my code for lines
    geo_data = gdf.__geo_interface__
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(fields=['route_id', 'val'],
                                               # aliases=tooltip_labels,
                                               labels=True,
                                               sticky=False)
    ).add_to(m)

    # adding a title
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                 '''.format(map_title)
    m.get_root().html.add_child(folium.Element(title_html))

    # adding a legend
    colorscale.caption = colorbar_str
    colorscale.add_to(m)

    return m

def create_map(trips_data, shapes, value, map_title,colorbar_str, window=None, mode='max'):
    """
    create a map of the routes that happen during the window color coded by value parameter
    :param trips_data: summarised trips dataframe
    :param shapes: shapes dataframe
    :param value: the value we wish to show in our plot, needs to be a list of the same length as trips_data, one value for each row
    :param window: optional list with two hour values specifying to only look at trips that start in that hour window
    :param mode: when aggregating multiple results from within a window, what mode to use 'max', 'mean', or 'min'
    :return:
    """
    gdf = _create_gdf_of_value(trips_data, shapes, value, window=window, mode=mode)
    m = _create_gdf_map(gdf, map_title, colorbar_str)
    return m

if __name__=="__main__":
    import RouteZero.bus as ebus
    from RouteZero.models import LinearRegressionAbdelatyModel

    trips_data = pd.read_csv('../data/gtfs/vic_metro_bus/trip_data.csv')
    trips_data['passengers'] = 38
    shapes = gpd.read_file('../data/gtfs/vic_metro_bus/shapes.shp')
    window = [5, 10]
    mode='max'

    bus = ebus.BYD()
    model = LinearRegressionAbdelatyModel()
    ec_km, ec_total = model.predict_hottest(trips_data, bus)

    value = ec_km
    gdf = _create_gdf_of_value(trips_data, shapes, ec_km, window=None, mode='max')

    map_title = "Route Energy Consumption"
    colorbar_str = 'energy per km'

    m = _create_gdf_map(gdf, map_title, colorbar_str)
    _folium_open(m, map_title+'.html')

