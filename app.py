import streamlit as st
import pandas as pd
import networkx as nx
from heapq import heappop, heappush
import os

# Helper functions for A* algorithm
def heuristic(a, b, graph):
    """Simple heuristic function for A* algorithm"""
    return 0  # For this implementation, we'll use 0 as a placeholder

def a_star(graph, start, goal):
    if start not in graph or goal not in graph:
        return [], -1
    
    open_set = []
    heappush(open_set, (0, start))
    came_from = {}
    g_score = {node: float('inf') for node in graph}
    f_score = {node: float('inf') for node in graph}
    g_score[start] = 0
    f_score[start] = heuristic(start, goal, graph)
    
    while open_set:
        current_f, current = heappop(open_set)
        
        if current == goal:
            path = []
            total_distance = g_score[goal]
            while current:
                path.append(current)
                current = came_from.get(current)
            return path[::-1], total_distance
        
        for neighbor, weight in graph[current]:
            tentative_g_score = g_score[current] + weight
            
            if tentative_g_score < g_score[neighbor]:
                came_from[neighbor] = current
                g_score[neighbor] = tentative_g_score
                f_score[neighbor] = g_score[neighbor] + heuristic(neighbor, goal, graph)
                heappush(open_set, (f_score[neighbor], neighbor))
    
    return [], -1

# Load graph from CSV file
def load_graph_from_csv(file):
    df = pd.read_csv(file)
    graph = {}
    for _, row in df.iterrows():
        city1, city2, distance = row['city1'], row['city2'], row['distance_between']
        
        # Ensure bidirectional connections
        graph.setdefault(city1, []).append((city2, distance))
        graph.setdefault(city2, []).append((city1, distance))
    
    return graph, df

# Streamlit app
def main():
    st.title("Best Path Finder")

    # Initialize session state for graph and file
    if 'graph' not in st.session_state:
        st.session_state.graph = {}
    if 'uploaded_file_name' not in st.session_state:
        st.session_state.uploaded_file_name = None
    if 'original_df' not in st.session_state:
        st.session_state.original_df = None

    # Upload CSV
    uploaded_file = st.file_uploader("Upload a CSV file containing cities and distances (city1, city2, Distance)", type=["csv"])
    
    if uploaded_file:
        # Check if it's a new file or the same file
        if st.session_state.uploaded_file_name != uploaded_file.name:
            # Load graph from uploaded file
            st.session_state.graph, st.session_state.original_df = load_graph_from_csv(uploaded_file)
            st.session_state.uploaded_file_name = uploaded_file.name
            st.success("Cities and distances loaded successfully!")

    # Display graph structure
    if st.checkbox("Display graph structure"):
        st.subheader("Graph Structure")
        if st.session_state.graph:
            for city, neighbors in st.session_state.graph.items():
                neighbors_str = ", ".join([f"{neighbor} ({dist} km)" for neighbor, dist in neighbors])
                st.write(f"{city} -> {neighbors_str}")
        else:
            st.write("No graph loaded. Please upload a CSV or add edges.")

    # Find shortest path
    st.subheader("Find Shortest Path")
    start_city = st.text_input("Enter starting city:")
    destination_city = st.text_input("Enter destination city:")
    
    if st.button("Find Path"):
        if start_city and destination_city:
            path, distance = a_star(st.session_state.graph, start_city, destination_city)
            if path:
                st.success(f"Shortest path: {' -> '.join(path)}")
                st.success(f"Total distance: {distance} km")
            else:
                st.error("Path not found!")
        else:
            st.error("Please enter both starting and destination cities.")

    # Add new edge
    st.subheader("Add New Edge")
    city1 = st.text_input("City 1:")
    city2 = st.text_input("City 2:")
    distance = st.number_input("Distance (in km):", min_value=0.0, format="%.2f")
    
    if st.button("Add Edge"):
        if city1 and city2 and distance > 0:
            if st.session_state.uploaded_file_name:
                # Ensure bidirectional connection in graph
                st.session_state.graph.setdefault(city1, []).append((city2, distance))
                st.session_state.graph.setdefault(city2, []).append((city1, distance))
                
                # Add to DataFrame
                new_row = pd.DataFrame({
                    'city1': [city1],
                    'city2': [city2],
                    'distance_between': [distance]
                })
                st.session_state.original_df = pd.concat([st.session_state.original_df, new_row], ignore_index=True)
                
                # Save to the original file
                st.session_state.original_df.to_csv(st.session_state.uploaded_file_name, index=False)
                
                st.success("Edge added successfully and saved to the original CSV!")
            else:
                st.error("Please upload a CSV file first.")
        else:
            st.error("Please fill out all fields correctly.")

# Run the Streamlit app
if __name__ == "__main__":
    main()
