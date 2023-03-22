import streamlit as st
import pandas as pd
import numpy as np
import requests
import folium
from streamlit_folium import folium_static
import math
import os
from sklearn.cluster import KMeans
from random import random
import plotly.graph_objs as go
import matplotlib.pyplot as plt
from sklearn.metrics import silhouette_score

# Add a title to the sidebar
st.sidebar.title("Welcome to my app!")

# Add a note to the sidebar
st.sidebar.write('''The slider below is used to select the number of clusters in Section 6 
where clustering is applied. By adjusting the slider, you can specify the desired 
number of groups to divide the addresses into.''')

# upload the google maps api key
if os.path.isfile("credentials.py"):
    import credentials
    google_API_KEY = credentials.google_API_KEY
else:
    google_API_KEY = st.secrets["google_API_KEY"]
    
# title and introduction
with st.container():
    st.title("Google Maps API Demo Streamlit App")
    st.write("Hello! This is Fatih. The purpose of this application is to explore the Google Maps API.")  

# user input and display                                       
with st.container():
    st.subheader("1. Enter Addresses")
    
    def add():
        if user_input:  # Check if user_input is not empty
            if "input_list" not in st.session_state:
                st.session_state.input_list = [user_input]
            else:
                st.session_state.input_list.append(user_input)
    
    def add_from_file(address_list):
        if "input_list" not in st.session_state:
            st.session_state.input_list = address_list
        else:
            st.session_state.input_list += address_list

    def select_all():
        if "input_list" in st.session_state:
            for i in range(len(st.session_state.input_list)):
                st.session_state[f"checkbox_{i}"] = True

    def unselect_all():
        if "input_list" in st.session_state:
            for i in range(len(st.session_state.input_list)):
                st.session_state[f"checkbox_{i}"] = False

    def display_addresses():
        # Display a checkbox for each input in the list
        if 'input_list' in st.session_state:
            for i, input_item in enumerate(st.session_state.input_list):
                st.checkbox(input_item, key=f"checkbox_{i}")  
    
    def get_selected_indexes():
        indexes = []
        if 'input_list' in st.session_state:
            for i, input_item in enumerate(st.session_state.input_list):
                if st.session_state[f"checkbox_{i}"]:
                    indexes.append(i)
        return indexes
    
    def remove_by_indexes(indexes, input_list):
        return [item for i, item in enumerate(input_list) if i not in indexes]

    def remove_selected():
        indexes = get_selected_indexes()
        if 'input_list' in st.session_state:
            st.session_state.input_list = remove_by_indexes(indexes, st.session_state.input_list)
        unselect_all()
    
    def find_matching_column(df, possible_names):
        for col in df.columns:
            if col in possible_names:
                return col
        return None
    
    # Take user input bulk (addresses)
    text1 = '''To add an address, type the address the box below and click 'Add Address'. 
    To add multiple addresses, do the same steps for each new one. 
    You can also bulk upload addresses with a CSV file.'''
    text2 = '''Choose a CSV file that has the addresses, one column should have 'Address' header. 
    Note: You can manually enter the addresses as well.'''
    
    st.write(text1)
    uploaded_file = st.file_uploader(text2, type="csv")
    
    if uploaded_file is not None:
        df = pd.read_csv(uploaded_file)
        
        # check if there is only one column in the dataframe
        if len(df.columns) == 1:
            address_col = df.columns[0]
        elif len(df.columns) > 1:
            possible_names = ['Address', 'Addresses', 'address', 'addresses']
            address_col = find_matching_column(df, possible_names)

        # convert the address column to a list
        if address_col:
            address_list = df[address_col].tolist()
        else:
            print("No address column found")

        add_from_file(address_list)
    
    # NOTE: Remove this else part later or use another sample data
    else:
        if "input_list" not in st.session_state:
            
            df = pd.read_csv('sample_addresses.csv')
            address_list = df['address'].tolist()
            add_from_file(address_list)
            st.warning("No file uploaded. Sample data used. Remove if desired.")

    # Take user input (addresses)
    user_input = st.text_input("Enter an address and click 'Add Address' button:")

    # Create the buttons in a horizontal layout
    cols = st.columns(4)
    with cols[0]:
        st.button('Add Address', on_click=add)
    with cols[1]:
        st.button('Select All', on_click=select_all)
    with cols[2]:
        st.button('Unselect All', on_click=unselect_all)
    with cols[3]:
        st.button('Remove Selected', on_click=remove_selected)

with st.expander("Click to show/hide the addresses", expanded=True):        
    # Display the addresses
    display_addresses()
        
# convert to coordinates and display as df        
with st.container():
    st.subheader("2. Converting Addresses to Geographic Coordinates")
    
    @st.cache_data
    def geocode_address(address, api_key):
        # URL encode the address
        encoded_address = requests.utils.quote(address)

        # Send a request to the Google Maps Geocoding API
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?address={encoded_address}&key={api_key}"
        response = requests.get(geocode_url)
        data = response.json()

        # Check the API response status and extract the coordinates
        if data['status'] == 'OK':
            lat = data['results'][0]['geometry']['location']['lat']
            lon = data['results'][0]['geometry']['location']['lng']
            return round(lat, 6), round(lon, 6)
        else:
            return None, None
    
    if 'input_list' in st.session_state:     
        lat_lng_list = []
        for address in st.session_state.input_list:
            latitude, longitude = geocode_address(address, google_API_KEY)
            if latitude == None:
                st.warning("Please enter a valid address. Remove the invalid address from the list.")
                break          
            # Add the address and latitude and longitude as a tuple to the list
            lat_lng_list.append((address, (latitude, longitude))) 
        
        # Create a pandas DataFrame with columns for the address and latitude and longitude tuples
        df = pd.DataFrame(lat_lng_list, columns=["address", "lat_lng"])
        # Display the latitude and longitude of each address to the user
        st.write(df)
        
# show locations on map        
with st.container():
    st.subheader("3. Show Addresses on Map")
    
    def create_map():
        # Create the map with Google Maps
        map_obj = folium.Map(tiles=None)
        folium.TileLayer("https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
                         attr="google", 
                         name="Google Maps", 
                         overlay=True, 
                         control=True, 
                         subdomains=["mt0", "mt1", "mt2", "mt3"]).add_to(map_obj)
        return map_obj
    
    def add_markers(map_obj, locations, popup_list=None):
        if popup_list is  None:
            # Add markers for each location in the DataFrame
            for lat, lon in locations:
                folium.Marker([lat, lon]).add_to(map_obj)
        else:
            for i in range(len(locations)):
                lat, lon = locations[i]
                popup = popup_list[i]
                folium.Marker([lat, lon], popup=popup).add_to(map_obj)

        # Fit the map bounds to include all markers
        south_west = [min(lat for lat, _ in locations) - 0.02, min(lon for _, lon in locations) - 0.02]
        north_east = [max(lat for lat, _ in locations) + 0.02, max(lon for _, lon in locations) + 0.02]
        map_bounds = [south_west, north_east]
        map_obj.fit_bounds(map_bounds)

        return map_obj

    if 'input_list' in st.session_state:
        if len(df) > 0:
            m = create_map()
            m = add_markers(m, df['lat_lng'], df['address'])
            folium_static(m)

# find the center, show distances            
with st.container():
    # Add a subheader
    st.subheader("4. Find the Center and Haversine Distances from Addresses")
    st.write("Haversine distance measures the shortest distance between two locations on the Earth's surface, accounting for its curvature.")
    
    # Print the instructions for sorting the table by distance
    st.write('''The 'distance_center_miles' column in the table shows the distances between each address and the center location, 
    measured in miles. To sort the addresses by distance, simply click on the header of the 'distance_center_miles' column.''')
    
    @st.cache_data
    def calculate_center_coords(coords_list):
        lat_mean = sum([coords[0] for coords in coords_list]) / len(coords_list)
        lng_mean = sum([coords[1] for coords in coords_list])  / len(coords_list)
        
        return [round(lat_mean, 6), round(lng_mean, 6)]
    
    @st.cache_data
    def reverse_geocode(lat, lon, api_key):
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
        response = requests.get(geocode_url)
        data = response.json()

        if data['status'] == 'OK':
            address = data['results'][0]['formatted_address']
        else:
            address = "Address not found"

        return address
    
    @st.cache_data
    def haversine_distance(lat1, lon1, lat2, lon2):
        # convert decimal degrees to radians
        lat1, lon1, lat2, lon2 = map(math.radians, [lat1, lon1, lat2, lon2])

        # haversine formula
        dlon = lon2 - lon1
        dlat = lat2 - lat1
        a = math.sin(dlat/2)**2 + math.cos(lat1) * math.cos(lat2) * math.sin(dlon/2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1-a))
        distance = 3956 * c  # radius of earth in miles

        return distance

    if 'input_list' in st.session_state:
        if len(df) > 0: 
            center_lat, center_lon = calculate_center_coords(df['lat_lng'])
            center_address = reverse_geocode(center_lat, center_lon, google_API_KEY)
            text = '**The center location is:** <i>{}</i>'.format(center_address)
            st.markdown(text, unsafe_allow_html=True)
            
            df['distance_center_miles'] = df['lat_lng'].apply(lambda x: haversine_distance(x[0], x[1], center_lat, center_lon))
            st.write(df)
            
# show center on the map            
with st.container():       
    st.subheader("5. Show Center of Locations on the Map")
    st.write("The center is the red pin on the map.")
    
    def add_center_marker(map_obj, lat, lon, color='red', icon='star', popup=None):
        new_marker = folium.Marker([lat, lon], icon=folium.Icon(color=color, icon=icon), popup=popup)
        map_obj.add_child(new_marker)
        
        return map_obj
    
    def add_lines_to_center(map_obj, locations_col, center_lat, center_lon, color='red', group_name='Lines'):
        line_group = folium.FeatureGroup(name=group_name)
        for lat_lng in locations_col:
            folium.PolyLine(locations=[lat_lng, [center_lat, center_lon]], color=color).add_to(line_group)
        map_obj.add_child(line_group)
        
        return map_obj
    
    if 'input_list' in st.session_state:
        if len(df) > 0:
    
            # Create the map and add markers
            m = create_map()
            m = add_markers(m, df['lat_lng'], df['address'])

            # Add a center marker
            center_lat, center_lon = calculate_center_coords(df['lat_lng'])
            center_address = reverse_geocode(center_lat, center_lon, google_API_KEY)
            m = add_center_marker(m, center_lat, center_lon, popup='This is the center: \n' + center_address)

            # Add lines from each location to the center location as a feature group
            m = add_lines_to_center(m, df['lat_lng'], center_lat, center_lon)

            # Add a layer control to toggle marker and line groups
            folium.LayerControl().add_to(m)

            text = '**The center location is:** <i>{}</i>'.format(center_address)
            st.markdown(text, unsafe_allow_html=True)

            st.write("Check the table above to see distances between each address and the center location")

            # Render the map in Streamlit
            folium_static(m)
    
    
# show center on the map            
with st.container():       
    st.subheader("6. Clustering Addresses Using K-Means Algorithm")
    
    st.write('''In this section, the addresses are clustered into groups using the K-Means algorithm. 
    You can specify the number of groups that you want to divide the addresses into by using slider below.''')
    
    st.write('''K-means is an unsupervised machine learning algorithm that divides data into clusters. 
    Its objective is to minimize the sum of squares of distances between data points and the center of the cluster they belong to.''')
    
    def cluster_latlng(coordinates_list, n_clusters):
        # Convert the latitude and longitude coordinates to a NumPy array
        X = np.array(coordinates_list)

        # Initialize the KMeans object with the number of clusters
        kmeans = KMeans(n_clusters=n_clusters, random_state=0)

        # Fit the KMeans model to the data
        kmeans.fit(X)

        # Get the cluster labels for each data point
        labels = kmeans.labels_

        # Get the cluster centroids
        centroids = kmeans.cluster_centers_

        # Return the cluster labels and centroids
        return labels, centroids
    
    def create_cluster_table(df, label_col):
    
        # Define the colorscale
        n = df[label_col].nunique()
        colorscale = []
        for _ in range(n):
            colorscale.append('rgba({},{},{},0.3)'.format(int(random()*255), 
                                                          int(random()*255), 
                                                          int(random()*255)))

        # Map the colors to the cluster labels
        colors = [colorscale[label] for label in df[label_col]]

        # Define the table data
        table_data = go.Table(
            header=dict(values=list(df.columns), fill_color='rgba(200, 200, 200, 1)'),
            cells=dict(values=[df[col] for col in df.columns], fill_color=[colors])
        )

        # Show the table
        fig = go.Figure(data=table_data)
        fig.update_layout(height=900)
        st.plotly_chart(fig)
        
    def miles_to_meters(miles):
        return miles * 1609.34
    
    def get_max_by_group(df, group_col, max_col):
        return df.groupby(group_col)[max_col].max()
    
    if 'input_list' in st.session_state:
        if len(df) > 0:
    
            # Define the default value for the slider
            default_value = 4

            # Set the default value to the length of the DataFrame if it's smaller
            if len(df) < default_value:
                default_value = len(df)
                
            # Initialize session state
            if "num_clusters" not in st.session_state:
                st.session_state.num_clusters = default_value
            
            # this is to fix sync issue
            rerun_flag = st.session_state.num_clusters
                
            # Define your sidebar slider
            num_clusters_sidebar = st.sidebar.slider("Select the number of clusters:", min_value=2, max_value=len(df), 
                                                     value=st.session_state.num_clusters,
                                                     key='num_clusters_sidebar')
            
            # Update session state with the sidebar slider value
            st.session_state.num_clusters = num_clusters_sidebar
            if st.session_state.num_clusters != rerun_flag:
                st.experimental_rerun()

            # Define the slider input with minimum and maximum values based on the DataFrame length
            num_clusters_main = st.slider('Select the number of clusters:', min_value=2, max_value=len(df), 
                                         value=st.session_state.num_clusters,
                                         key='num_clusters_main')
            # Update session state with the sidebar slider value
            st.session_state.num_clusters = num_clusters_main
            if st.session_state.num_clusters != num_clusters_sidebar:
                st.experimental_rerun()
                

            # run the model
            labels, centroids = cluster_latlng(df['lat_lng'].to_list(), num_clusters_main)
            
            # Create a success message
            st.success('Clustering complete! View the results below. Number of clusters: {}'.format(num_clusters_main))
            
            df['cluster_label'] = labels
            df['cluster_center'] = df['cluster_label'].apply(lambda x: [round(centroids[x][0], 6), round(centroids[x][1], 6)])
            df = df.sort_values('cluster_label')
            
            # Show clustered table
            create_cluster_table(df, 'cluster_label')
            
            # Calculate the distances from cluster centers
            df['distance_cluster_center_miles'] = df.apply(lambda x: haversine_distance(x['lat_lng'][0], 
                                                                            x['lat_lng'][1], 
                                                                            x['cluster_center'][0], 
                                                                            x['cluster_center'][1]), axis=1)
            
            st.write(df[['address', 'cluster_label', 'distance_cluster_center_miles']])
            
            max_distances = get_max_by_group(df, 'cluster_label', 'distance_cluster_center_miles')
            
            # Map with Circles
            m = create_map()
            m = add_markers(m, df['lat_lng'], df['address'])
            
            # Add cluster center
            for i in range(len(centroids)):
                c = centroids[i]
                add_center_marker(m, c[0], c[1], popup='Cluster {} Center Point'.format(i))
            
            # Add circle markers for each centroid
            for i in range(len(centroids)):
                center = centroids[i]
                radius = miles_to_meters(max_distances[i]) + 1000
                folium.Circle(location=center, radius=radius, color='red', fill_color='red', fill_opacity=0.2).add_to(m)
            
            folium_static(m)
            
            
            # Map with Lines
            m = create_map()
            m = add_markers(m, df['lat_lng'], df['address'])
            
            # Add cluster center
            for i in range(len(centroids)):
                c = centroids[i]
                add_center_marker(m, c[0], c[1], popup='Cluster {} Center Point'.format(i))
                
            # Loop through the data and add lines from each address to its cluster center
            for i in range(len(df)):
                # Get the coordinates for the address and its cluster center
                address_coords = df.loc[i, 'lat_lng']
                center_coords = centroids[df.loc[i, 'cluster_label']]
    
                # Add a line from the address to its cluster center
                folium.PolyLine(locations=[address_coords, center_coords], color='red').add_to(m)
        
            folium_static(m)
            
# show center on the map            
with st.container():       
    st.subheader("7. What is the optimal number of clusters?")
    
    st.write('''Selecting the optimal number of clusters is crucial in K-means clustering, as it determines the structure of the resulting groups. 
    The right number of clusters helps to ensure that the groups are meaningful and accurately reflect the patterns in the data. 
    To determine the optimal number of clusters, two commonly used methods are used; the elbow method and silhouette score method.''')
    
    st.markdown("#### Elbow method")
    st.write('''The identification of the elbow point in a plot is subjective, and there is no specific rule for its definition. 
    Typically, it is defined as the point on the curve where the rate of decrease in WSS starts to level off, forming an "elbow" shape. 
    One way to identify it is to look for the point where the rate of decrease in WSS significantly slows down, forming an elbow-like shape. 
    Another approach is to draw a line from the first point to the last point on the curve and identify the point 
    on the curve farthest from this line, representing a significant change in the rate of decrease in WSS and considered the elbow point.''')

    def plot_elbow_method(coordinates_list):
        # Load the dataset
        X = np.array(coordinates_list)

        # Define a range of k values to try
        if len(coordinates_list) < 10:
            n = len(coordinates_list)
        else:
            n = 10
        k_values = range(1, n)

        # Fit KMeans models for each k value and calculate the WSS
        wss_values = []
        for k in k_values:
            model = KMeans(n_clusters=k).fit(X)
            wss = model.inertia_
            wss_values.append(wss)

        # Plot the WSS values against the number of clusters
        fig, ax = plt.subplots()
        ax.plot(k_values, wss_values, 'bx-')
        ax.set_xlabel('Number of clusters (k)')
        ax.set_ylabel('Within-cluster sum of squares (WSS)')
        ax.set_title('Elbow method for optimal k')

        # Draw a line from the first point to the last point
        line_x = [1, k_values[-1]]
        line_y = [wss_values[0], wss_values[-1]]
        ax.plot(line_x, line_y, 'r--')

        # Find the point farthest away from the line
        distances = []
        for i in range(len(k_values)):
            x = k_values[i]
            y = wss_values[i]
            distance = abs((line_y[1] - line_y[0]) * x - (line_x[1] - line_x[0]) * y + 
                           line_x[1] * line_y[0] - line_y[1] * line_x[0]) / ((line_y[1] - line_y[0]) ** 2 + (line_x[1] - line_x[0]) ** 2) ** 0.5
            distances.append(distance)
        elbow_point = k_values[distances.index(max(distances))]

        # Highlight the elbow point
        ax.plot(elbow_point, wss_values[elbow_point - 1], 'ro')

        # Display the plot in Streamlit
        st.pyplot(fig)
    plot_elbow_method(df['lat_lng'].to_list())
    
    st.markdown("#### Silhouette score method")
    st.write('''The silhouette score measures how well each data point in a dataset is matched to its own cluster compared to other clusters. 
    It ranges from -1 to 1, with a higher score indicating better clustering. The score is computed for each data point, 
    and the mean score across all data points is used to evaluate the clustering solution.''')
    
    def get_silhouette_scores(coordinates_list):
        # Load the dataset
        coordinates_list = coordinates_list
        X = np.array(coordinates_list)

        # Define a range of k values to try
        k_values = range(2, len(df))

        # Compute the silhouette score for each value of k
        silhouette_scores = []
        for k in k_values:
            model = KMeans(n_clusters=k)
            labels = model.fit_predict(X)
            silhouette = silhouette_score(X, labels)
            silhouette_scores.append(silhouette)
        return silhouette_scores

    silhouette_scores = get_silhouette_scores(df['lat_lng'].to_list())
    
    st.success("Silhouette scores are calculated for different values of k (the number of clusters)!")
    df_silhouette_scores = pd.DataFrame(silhouette_scores, columns=['Silhouette Scores'])
    df_silhouette_scores['Number of clusters'] = range(2, len(df))
    st.table(df_silhouette_scores)