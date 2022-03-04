import numpy as np
import pandas as pd
pd.options.display.max_columns = None
from scipy.interpolate import interp1d
from gtfs_routes import process_gtfs_routes
from weather import location_design_temp

# for plotting the demo example
import matplotlib.pyplot as plt
import geopandas as gpd
import folium
import webbrowser
import branca

def folium_open(f_map, path):
    html_page = f'{path}'
    f_map.save(html_page)
    # open in browser.
    new = 2
    webbrowser.open(html_page, new=new)


class LinearRegressionAbdelatyModel:
    """
    A linear regression model for eBus energy consumption taken from

    Abdelaty, H.; Mohamed, M. APrediction Model for Battery Electric Bus Energy Consumption in Transit.
    Energies 2021, 14, 2824. https://doi.org/10.3390/en14102824

    model:

    EC = B0 + B1*GR + B2*SoCi + B3*RC + B4*HVAC + B5*PL + B6*Dagg + B7*SD + B8*Va + e

    Model is then altered slightly so that hvac is factored off time not distance

    where:
    EC: energy consumption (KWh/km)
    GR: road grade average (%) [-100, 100]
    SoCi: initial battery state of charge (%) [0, 100]
    RC: road condition in three levels (three levels)
    HVAC: auxilliary systems (heating, cooling etc) (KW)
    PL: passenger loading (passengers)
    Dagg: driver aggressiveness (three levels)
    SD: stop density (stops/km)
    Va: average velocity (km/h)

    Parameter values are given as
    B0 = -0.782
    B1 = 0.38
    B2 = 0.0124
    B3 = 0.26
    B4 = 0.036
    B5 = 0.005
    B6 = 0.065
    B7 = 0.128
    B8 = 0.007
    """
    def __init__(self):
        # self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.036, 0.005, 0.065, 0.128, 0.007])
        self.B = np.array([-0.782, 0.38, 0.0124, 0.26, 0.005, 0.065, 0.128, 0.007])
        self.hvac_ref_vals = [13.75, 6.7, 3.0, 1.25, 2.0, 10.75]
        self.temp_ref_vals = [-20, -10, 0, 10, 20, 30]
        self.hvac_func = interp1d(self.temp_ref_vals, self.hvac_ref_vals, kind='cubic', fill_value='extrapolate', bounds_error=False)

    def _predict(self, X):
        return np.dot(X, self.B)

    # todo: extend the below so can take lists for the optional inputs
    def predict_routes(self, route_data, SoCi=1., Dagg=2., PL=38, RC=1.):
        route_data['min_EC_km'] = 0.
        route_data['max_EC_km'] = 0.
        route_data['min_EC_total'] = 0.
        route_data['max_EC_total'] =0.
        for i, r in route_data.iterrows():
            hvac = []
            hvac.append(self.hvac_from_temp(r['max_temp'], 1/(r['average_speed_mps']*3.6)))
            hvac.append(self.hvac_from_temp(r['min_temp'], 1/(r['average_speed_mps']*3.6)))
            if (10. < r['max_temp']) and (10. > r['min_temp']):
                hvac.append(self.hvac_from_temp(10., 1/(r['average_speed_mps']*3.6)))
            hvac_min = np.min(hvac)
            hvac_max = np.max(hvac)
            X = [1., r['av_grade_%'], SoCi, RC, PL, Dagg, r['stops_km'], r['average_speed_mps'] * 3.6]
            EC_min = self._predict(X) + hvac_min
            EC_max = self._predict(X) + hvac_max
            route_data.loc[i, 'min_EC_km'] = EC_min
            route_data.loc[i, 'max_EC_km'] = EC_max
            route_data.loc[i, 'min_EC_total'] = EC_min * r['trip_distance'] / 1000  # convert metres to km
            route_data.loc[i, 'max_EC_total'] = EC_max * r['trip_distance'] / 1000
        return route_data

    def hvac_from_temp(self, temps, hours_km):
        hvacs = self.hvac_func(temps) * hours_km
        return hvacs

if __name__ == "__main__":

    gtfs_file = "./data/full_greater_sydney_gtfs_static.zip"        # location of the gtfs zip file
    # route_short_names = ["305", "320"]      # the short names of the routes we want to get summaries of
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433',
    #                      '437', '438N', '438X', '440',
    #                      '441', '442', '445', '469',
    #                      '470', '502', '503', '504']      # the short names of the routes we want to get summaries of
    # route_short_names = ["305", "320", '389', '406',
    #                      '428', '430', '431', '433',
    #                      '437', '438N', '438X', '440',
    #                      '441', '442', '445', '469',
    #                      '470', '502', '503', '504',
    #                      '504X', '']  # the short names of the routes we want to get summaries of
    route_names_df = pd.read_csv('data/zenobe_routes.csv')
    route_short_names = route_names_df['route_short_name'].to_list()

    # route_desc = 'Sydney Buses Network'     # optional input if we also want to filter by particular types of routes
    route_desc = None
    # cutoffs = [0, 6, 9, 15, 19, 22, 24]     # optional input for splitting up the route summary information into time windows
    passenger_loading = 38
    print('Processing routes '+", ".join(route_short_names))
    route_data, subset_shapes, elevation_profiles, trip_totals, subset_stops = process_gtfs_routes(gtfs_file, route_short_names, cutoffs=None, busiest_day=True, route_desc=route_desc)

    # get design day temperatures for route locations
    min_temps = []
    max_temps = []
    # avg_temps = []
    for i, r in route_data.iterrows():
        location_coords = (r['start_location'][1], r['start_location'][0] ) # geometry locations are (E, N) not (N, E)...
        elevation = elevation_profiles[r['shape_id']].mean()
        min_temp, max_temp, avg_temp = location_design_temp(location_coords, elevation, num_years=10, percentiles=[1, 99])
        min_temps.append(min_temp)
        max_temps.append(max_temp)
        # avg_temps.append(avg_temp)

    route_data['min_temp'] = min_temps
    route_data['max_temp'] = max_temps

    model = LinearRegressionAbdelatyModel()
    route_data = model.predict_routes(route_data, PL=passenger_loading)

    ## Prepare data for plotting on a map
    gdf = gpd.GeoDataFrame(subset_shapes)
    gdf['route_id'] = "nan"
    gdf['max_EC_total'] = np.nan
    gdf['max_EC_km'] = np.nan

    for i, s in gdf.iterrows():
        shape_id = s['shape_id']
        max_EC_total = route_data[route_data.shape_id==shape_id].max_EC_total
        max_EC_km = route_data[route_data.shape_id == shape_id].max_EC_km
        route_id = route_data[route_data.shape_id == shape_id].route_id.to_numpy()[0]
        if len(max_EC_total):
            gdf.at[i, 'max_EC_total'] = max_EC_total
            gdf.at[i, 'max_EC_km'] = max_EC_km
            gdf.at[i, 'route_id'] = route_id

    gdf.dropna(inplace=True)

    minx, miny, maxx, maxy = gdf.geometry.total_bounds

    centroid_lat = miny + (maxy - miny) / 2
    centroid_lon = minx + (maxx - minx) / 2

    ## create a map of total energy consumption
    m = folium.Map(location=[centroid_lat, centroid_lon],
                   tiles='cartodbpositron', zoom_start=12)
    gdf.crs = {'init': 'epsg:4326'}

    colorscale = branca.colormap.linear.YlGnBu_09.scale(gdf['max_EC_total'].min(), gdf['max_EC_total'].max())


    def style_function(feature):
        return {
            'fillOpacity': 0.5,
            'weight': 3,  # math.log2(feature['properties']['speed'])*2,
            'color': colorscale(feature['properties']['max_EC_total'])
        }


    # my code for lines
    geo_data = gdf.__geo_interface__
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(fields=['route_id', 'max_EC_total'],
                                               # aliases=tooltip_labels,
                                               labels=True,
                                               sticky=False)
    ).add_to(m)

    # adding a title
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                 '''.format("Route Energy Consumption")
    m.get_root().html.add_child(folium.Element(title_html))

    # adding a legend
    colorscale.caption = 'Energy consumption (kwh)'
    colorscale.add_to(m)

    folium_open(m, 'test.html')

    ## create a map of total energy consumption per km
    m = folium.Map(location=[centroid_lat, centroid_lon],
                   tiles='cartodbpositron', zoom_start=12)
    gdf.crs = {'init': 'epsg:4326'}

    colorscale = branca.colormap.linear.YlGnBu_09.scale(gdf['max_EC_km'].min(), gdf['max_EC_km'].max())


    def style_function(feature):
        return {
            'fillOpacity': 0.5,
            'weight': 3,  # math.log2(feature['properties']['speed'])*2,
            'color': colorscale(feature['properties']['max_EC_km'])
        }


    # my code for lines
    geo_data = gdf.__geo_interface__
    folium.GeoJson(
        geo_data,
        style_function=style_function,
        tooltip=folium.features.GeoJsonTooltip(fields=['route_id', 'max_EC_km'],
                                               # aliases=tooltip_labels,
                                               labels=True,
                                               sticky=False)
    ).add_to(m)

    # adding a title
    title_html = '''
                 <h3 align="center" style="font-size:16px"><b>{}</b></h3>
                 '''.format("Route Energy Consumption per km")
    m.get_root().html.add_child(folium.Element(title_html))

    # adding a legend
    colorscale.caption = 'Energy consumption per km (khw/km)'
    colorscale.add_to(m)

    folium_open(m, 'test.html')

    plt.subplot(2,1,1)
    plt.hist(route_data['max_EC_total'], bins=np.arange(40))
    plt.xlabel('Energy consumed on routes (kwh)')
    plt.ylabel('number of trips')
    plt.title('Energy consumptions for worst case temperature')

    plt.subplot(2,1,2)
    plt.hist(route_data['min_EC_total'], bins=np.arange(40))
    plt.xlabel('Energy consumed on routes (kwh)')
    plt.ylabel('number of trips')
    plt.title('Energy consumptions for best case temperature')

    plt.tight_layout()
    plt.show()

    #
    # subset_shapes.to_csv('pre_processed_gtfs_data/subset_shapes.csv')
    # route_data.to_csv('route_data.csv')

    # import pickle
    # with open('elevation_profiles.pickle', 'wb') as handle:
    #     pickle.dump(elevation_profiles, handle, protocol=pickle.HIGHEST_PROTOCOL)

    # with open('filename.pickle', 'rb') as handle:
    #     b = pickle.load(handle)
