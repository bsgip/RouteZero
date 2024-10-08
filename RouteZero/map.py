"""

    RouteZero module with functions for creating the route energy usage map

"""

import geopandas as gpd
import pandas as pd
import branca
import folium
import webbrowser

def _folium_open(f_map, path):
    "Opens a folium html webpage in browser"
    html_page = f'{path}'
    f_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)


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
        tmp['window_strings'] = tmp['hour window'].apply(lambda x:"{f_hour}:{f_min:02d} - {e_hour}:{e_min:02d}".format(f_hour=int(x[0]), f_min=int((x[0]*60) % 60),
                                                                 e_hour=int(x[1]), e_min=int((x[1]*60) % 60)))
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

    if "route_long_name" in gdf.columns:
        gdf['route_long_name'] = gdf['route_long_name'].fillna('')

    gdf.rename(columns={'average gradient (%)':'gradient (%)',
                        "average speed (km/h)":"speed (km/h)",
                        "possible max temp":"max temp",
                        "possible min temp":"min temp",
                        "trip distance (m)":"length (m)",
                        "route_long_name":"route long name",
                        'route_short_name':'route short name'}, inplace=True)
    gdf.dropna(inplace=True)


    return gdf

def _create_gdf_map(gdf, map_title, colorbar_str, min_val=None, max_val=None, total=False):
    """
    create map showing information contained in the geopandas dataframe
    :param gdf: geopandas dataframe containing the route information to be plotted
    :param map_title: map title string
    :param colorbar_str: string for the colorbar
    :param min_val: colorbar minimum value
    :param max_val: colorbar maximumvalue
    :param total: if true, displays the total energy consumption, if false, displays the energy consumption per km
    :return m: a folium map object
    """
    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    centroid_lat = miny + (maxy - miny) / 2
    centroid_lon = minx + (maxx - minx) / 2

    if total:
        if min_val is None:
            min_val = gdf['ec (kwh)'].min()
        if max_val is None:
            max_val = gdf['ec (kwh)'].max()
    else:
        if min_val is None:
            min_val = gdf['ec/km (kwh/km)'].min()
        if max_val is None:
            max_val = gdf['ec/km (kwh/km)'].max()

    ## create a map of total energy consumption
    m = folium.Map(location=[centroid_lat, centroid_lon],
                   tiles='cartodbpositron', zoom_start=12)
    gdf.crs = {'init': 'epsg:4326'}

    # colorscale = branca.colormap.linear.YlGnBu_09.scale(min_val, max_val)
    colorscale = branca.colormap.linear.viridis.scale(min_val, max_val)

    def style_function(feature):
        if total:
            return {
                'fillOpacity': 0.5,
                'weight': 3,  # math.log2(feature['properties']['speed'])*2,
                'color': colorscale(feature['properties']['ec (kwh)']),
                # "dashArray": '20, 20'
            }
        else:
            return {
                'fillOpacity': 0.5,
                'weight': 3,  # math.log2(feature['properties']['speed'])*2,
                'color': colorscale(feature['properties']['ec/km (kwh/km)']),
                # "dashArray": '20, 20'
            }

    geo_data = gdf.__geo_interface__
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(fields=['route short name','route_id', 'ec/km (kwh/km)', 'gradient (%)',
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

def create_map(route_summaries, shapes, map_title,colorbar_str, window=None, total=False):
    """
    create a map of the routes that happen during the window color coded by value parameter
    :param route_summaries: summarised route results
    :param shapes: shapes dataframe
    :param window: optional list with two hour values specifying to only look at trips that start in that hour window
    :return:
    """
    if total:
        max_val = route_summaries['ec (kwh)'].max()
        min_val = route_summaries['ec (kwh)'].min()
    else:
        max_val = route_summaries['ec/km (kwh/km)'].max()
        min_val = route_summaries['ec/km (kwh/km)'].min()

    gdf = _create_gdf_of_value(route_summaries, shapes, window=window)
    max_temp = gdf["max temp"].max()
    min_temp = gdf["min temp"].min()
    map_title = map_title + " allowing for\n a maximum temperature on routes of {:.1f} and a minimum temperature on " \
                            "routes of {:.1f}".format(max_temp, min_temp)
    m = _create_gdf_map(gdf, map_title, colorbar_str, max_val=max_val, min_val=min_val, total=total)
    return m

if __name__=="__main__":
    import RouteZero.bus as ebus
    from RouteZero.models import PredictionPipe, summarise_results

    name = 'Tas_hobart'

    trips_data = pd.read_csv('../data/gtfs/'+name+'/trip_data.csv')
    trips_data['passengers'] = 38

    shape_ids = trips_data['shape_id'].astype('str')

    shapes = gpd.read_file('../data/gtfs/'+name+'/shapes.shp')
    window = [5, 20]
    mode='max'

    bus = ebus.BYD()
    # model = LinearRegressionAbdelatyModel()
    prediction_pipe = PredictionPipe()
    ec_km, ec_total = prediction_pipe.predict_worst_case(trips_data, bus)

    route_summaries = summarise_results(trips_data, ec_km, ec_total)

    gdf = _create_gdf_of_value(route_summaries, shapes, window='6:00 - 9:30')

    map_title = "Route Energy Consumption"
    colorbar_str = 'energy per km'

    m = _create_gdf_map(gdf, map_title, colorbar_str)
    _folium_open(m, map_title+'.html')

