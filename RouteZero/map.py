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

# def route_energy_map(trip_data, energy_consumption, shapes, window=None, mode='max'):
#
#     tmp = trip_data.copy()
#     tmp['EC'] = energy_consumption
#
#     # if window is not None:
#     #     tmp = tmp[tmp['trip_start_time'] > ]
#
#     ## Prepare data for plotting on a map
#     shape_ids = trip_data.shape_id
#
#     gdf = gpd.GeoDataFrame(shapes)
#     gdf['route_id'] = "nan"
#     gdf['max_EC_total'] = np.nan
#     gdf['max_EC_km'] = np.nan

def _create_gdf_of_value(route_summaries, shapes, window=None):
    """
    creates a geopandas dataframe that has route shapes and an associated value so that routes can be plotted on a map
    and color coded by value
    :param route_summaries: summaries of the routes binned into time windows
    :param shapes: shapes dataframe
    :param window: optional list with two hour values specifying to only look at trips that start in that hour window
    :return:
    """

    tmp = route_summaries.copy()


    tmp.reset_index(inplace=True)       # fixes indexing bug

    # filter to specified window
    if window is not None:
        tmp['window_strings'] = tmp['hour window'].apply(lambda x:"{one:.1f} - {two:.1f}".format(one=x[0], two=x[1]))
        tmp = tmp[tmp['window_strings']==window]
        tmp.reset_index(drop=True, inplace=True)

    inds = tmp.groupby(by=['shape_id','route_short_name','route_id','direction_id'])['ec/km (kwh/km)'].idxmax().values
    filtered=tmp.iloc[inds]
    shape_ids = filtered['shape_id'].astype('str')
    filtered['shape_id'] = filtered['shape_id'].astype('str')

    ## Prepare data for plotting on a map
    gdf = gpd.GeoDataFrame(shapes[shapes['shape_id'].isin(shape_ids.unique().tolist())])
    gdf = gdf.merge(filtered, how='left')

    if 'length (m)' in gdf.columns:
        gdf.drop(columns=['length (m)'], inplace=True)

    gdf.rename(columns={'average gradient (%)':'gradient (%)',
                        "average speed (km/h)":"speed (km/h)",
                        "possible max temp":"max temp",
                        "possible min temp":"min temp",
                        "trip distance (m)":"length (m)"}, inplace=True)
    gdf.dropna(inplace=True)


    return gdf

def _create_gdf_map(gdf, map_title, colorbar_str, min_val=None, max_val=None):
    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    centroid_lat = miny + (maxy - miny) / 2
    centroid_lon = minx + (maxx - minx) / 2

    if min_val is None:
        min_val = gdf['ec/km (kwh/km)'].min()
    if max_val is None:
        max_val = gdf['ec/km (kwh/km)'].max()

    ## create a map of total energy consumption
    m = folium.Map(location=[centroid_lat, centroid_lon],
                   tiles='cartodbpositron', zoom_start=10)
    gdf.crs = {'init': 'epsg:4326'}

    # colorscale = branca.colormap.linear.YlGnBu_09.scale(min_val, max_val)
    colorscale = branca.colormap.linear.viridis.scale(min_val, max_val)

    def style_function(feature):
        return {
            'fillOpacity': 0.5,
            'weight': 3,  # math.log2(feature['properties']['speed'])*2,
            'color': colorscale(feature['properties']['ec/km (kwh/km)']),
            # "dashArray": '20, 20'
        }

    # my code for lines
    geo_data = gdf.__geo_interface__
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(fields=['route_id', 'ec/km (kwh/km)', 'gradient (%)',
                                                       'stops/km', "speed (km/h)", "max temp", "min temp",
                                                       'length (m)', 'ec (kwh)'],
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

def create_map(route_summaries, shapes, map_title,colorbar_str, window=None):
    """
    create a map of the routes that happen during the window color coded by value parameter
    :param route_summaries: summarised route results
    :param shapes: shapes dataframe
    :param window: optional list with two hour values specifying to only look at trips that start in that hour window
    :return:
    """

    max_val = route_summaries['ec/km (kwh/km)'].max()
    min_val = route_summaries['ec/km (kwh/km)'].min()

    gdf = _create_gdf_of_value(route_summaries, shapes, window=window)
    m = _create_gdf_map(gdf, map_title, colorbar_str, max_val=max_val, min_val=min_val)
    return m

if __name__=="__main__":
    import RouteZero.bus as ebus
    from RouteZero.models import LinearRegressionAbdelatyModel, summarise_results

    trips_data = pd.read_csv('../data/gtfs/adelaide/trip_data.csv')
    trips_data['passengers'] = 38

    shape_ids = trips_data['shape_id'].astype('str')

    shapes = gpd.read_file('../data/gtfs/adelaide/shapes.shp')
    window = [5, 10]
    mode='max'

    bus = ebus.BYD()
    model = LinearRegressionAbdelatyModel()
    ec_km, ec_total = model.predict_hottest(trips_data, bus)

    route_summaries = summarise_results(trips_data, ec_km, ec_total)

    gdf = _create_gdf_of_value(route_summaries, shapes, window='6.0 - 9.5')

    map_title = "Route Energy Consumption"
    colorbar_str = 'energy per km'

    m = _create_gdf_map(gdf, map_title, colorbar_str)
    _folium_open(m, map_title+'.html')

