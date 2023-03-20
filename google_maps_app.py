import streamlit as st
import pandas as pd
import requests
import folium
from folium.plugins import MiniMap
from streamlit_folium import folium_static
import math


import credentials
google_API_KEY = credentials.google_API_KEY


with st.container():
    # Add a title to the app
    st.title("Google Maps API Demo Streamlit App")
    st.write("Hello! This is Fatih. Welcome to my App. The purpose of this application is to explore the Google Maps API.")  

                                                     
with st.container():
    # Add a subheader
    st.subheader("Step 1: Enter Addresses")
    
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
    text1 = "To add an address, type the address the box below and click 'Add Address'. To add multiple addresses, do the same steps for each new one. You can also bulk upload addresses with a CSV file."
    text2 = "Choose a CSV file that has the addresses, one column should have 'Address' header. Note: You can manually enter the addresses as well."
    
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
    
    # Display the addresses
    display_addresses()
        
        
with st.container():
    # Add a subheader
    st.subheader("Step 2: Converting Addresses to Geographic Coordinates")
    
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
        
        
with st.container():
    # Add a subheader
    st.subheader("Step 3: Show Addresses on Map")
    
    def find_center(locations):
        center_lat = sum(lat for lat, _ in locations) / len(locations)
        center_lon = sum(lon for _, lon in locations) / len(locations)
        return center_lat, center_lon
    
    def plot_map(locations, center_lat, center_lon):
        # Calculate the map bounds
        south_west = [min(lat for lat, _ in locations) - 0.04, min(lon for _, lon in locations) - 0.04]
        north_east = [max(lat for lat, _ in locations) + 0.04, max(lon for _, lon in locations) + 0.04]
        map_bounds = [south_west, north_east]

        # Create the map with Google Maps
        m = folium.Map(tiles=None)
        folium.TileLayer("https://{s}.google.com/vt/lyrs=m&x={x}&y={y}&z={z}", 
                         attr="google", 
                         name="Google Maps", 
                         overlay=True, 
                         control=True, 
                         subdomains=["mt0", "mt1", "mt2", "mt3"], 
                         api_key=google_API_KEY).add_to(m)

        # Add markers for each location in the list
        for lat, lon in locations:
            folium.Marker([lat, lon]).add_to(m)
            
        folium.Marker([center_lat, center_lon], 
                      icon=folium.Icon(color="red", icon=""), 
                      popup="Center of the locations").add_to(m)

        # # Add a minimap
        # minimap = MiniMap()
        # m.add_child(minimap)
        
        # Fit the map bounds to include all markers
        m.fit_bounds(map_bounds)

        # Display the map
        folium_static(m)
    
    if 'input_list' in st.session_state:
        if len(df) > 0:
            locations = df["lat_lng"]
            center_lat, center_lon = find_center(locations)
            plot_map(locations, center_lat, center_lon)
      
    
with st.container():
    # Add a subheader
    st.subheader("Step 4: Find the Center and Haversine Distances from Addresses")
    
    def reverse_geocode(lat, lon, api_key):
        geocode_url = f"https://maps.googleapis.com/maps/api/geocode/json?latlng={lat},{lon}&key={api_key}"
        response = requests.get(geocode_url)
        data = response.json()

        if data['status'] == 'OK':
            address = data['results'][0]['formatted_address']
        else:
            address = "Address not found"

        return address
    
    def haversine_distance(lat1, lon1, lat2, lon2):
        # Convert latitude and longitude from degrees to radians
        lat1_rad = math.radians(lat1)
        lon1_rad = math.radians(lon1)
        lat2_rad = math.radians(lat2)
        lon2_rad = math.radians(lon2)

        # Calculate the differences between latitudes and longitudes
        delta_lat = lat2_rad - lat1_rad
        delta_lon = lon2_rad - lon1_rad

        # Calculate the Haversine formula
        a = math.sin(delta_lat / 2)**2 + math.cos(lat1_rad) * math.cos(lat2_rad) * math.sin(delta_lon / 2)**2
        c = 2 * math.atan2(math.sqrt(a), math.sqrt(1 - a))

        # Convert the angular distance to miles
        earth_radius_miles = 3958.8
        distance_miles = earth_radius_miles * c

        return distance_miles
    
    def calculate_distance(row, center_lat, center_lon):
        lat1, lon1 = row['lat_lng']
        return haversine_distance(lat1, lon1, center_lat, center_lon)

    if 'input_list' in st.session_state:
        if len(df) > 0: 
            center_address = reverse_geocode(center_lat, center_lon, google_API_KEY)
            st.write("The center of the address is:")
            st.write(center_address)
            st.write("Note: The center is the red pin on the map.")
               
            df['distance_to_center'] = df.apply(calculate_distance, args=(center_lat, center_lon), axis=1)
            df['unit'] = 'miles'
            st.write(df)