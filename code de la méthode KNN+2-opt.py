# KNN + 2-opt implementation for VRP
import math
import matplotlib.pyplot as plt
from collections import deque
import pandas as pd

# Configuration des paramètres
N_CAMIONS = 3
CAPACITE = 8000
K_NEIGHBORS = 5  # Nombre de voisins à considérer

def load_data_from_excel(file_path):
    """Extraction des données depuis le fichier Excel"""
    try:
        df = pd.read_excel(file_path)
        print("Column names in Excel file:", list(df.columns))
        print("Data:\n", df)
        
        # Définir les flags actifs originaux pour correspondre à votre référence
        original_active = {
            1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 10: 1,
            11: 0, 12: 0, 13: 1, 14: 0, 15: 0, 16: 0, 17: 1, 18: 0, 19: 0, 20: 1
        }
        
        data = {0: {"x": 0, "y": 0, "q": 0, "active": 0}}  # Dépôt
        df_transposed = df.drop('ID', axis=1).T
        
        for col in df_transposed.index:
            location_data = df_transposed.loc[col]
            data[int(col)] = {
                'x': float(location_data[0]),  # Coordonnée X (non mise à l'échelle)
                'y': float(location_data[1]),  # Coordonnée Y (non mise à l'échelle)
                'q': float(location_data[2]),  # Quantité
                'active': original_active[int(col)]  # Statut actif
            }
        
        print("\nLoaded data from Excel (unscaled coordinates):")
        for key, value in data.items():
            print(f"ID {key}: {value}")
        
        return data
    
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        raise

def greedy_knn_vrp(data, capacity, k_neighbors):
    """Implémentation de l'algorithme Greedy k-Nearest Neighbors pour VRP"""
    active_nodes = [n for n in data if data[n]['active'] == 1 and n != 0]
    if not active_nodes:
        raise ValueError("Aucun bac actif à collecter!")
   
    # Tri initial par distance au dépôt
    active_nodes.sort(key=lambda n: math.hypot(data[n]['x'], data[n]['y']))
   
    routes = []
    remaining_capacity = [capacity] * N_CAMIONS
    visited = set()
   
    for truck in range(N_CAMIONS):
        if not active_nodes:
            break
           
        current_route = [0]
        current_load = 0
        current_position = 0  # Dépôt
       
        while True:
            # Trouver les k plus proches voisins non visités
            neighbors = []
            for node in active_nodes:
                if node not in visited:
                    dist = math.hypot(data[node]['x'] - data[current_position]['x'],
                                      data[node]['y'] - data[current_position]['y'])
                    neighbors.append((dist, node))
           
            if not neighbors:
                break
               
            # Sélectionner les k meilleurs voisins valides
            neighbors.sort()
            candidates = neighbors[:k_neighbors]
            found = False
           
            for dist, candidate in candidates:
                if (current_load + data[candidate]['q'] <= remaining_capacity[truck]):
                    current_route.append(candidate)
                    current_load += data[candidate]['q']
                    visited.add(candidate)
                    active_nodes.remove(candidate)
                    current_position = candidate
                    found = True
                    break
                   
            if not found:
                break
               
        if len(current_route) > 1:
            current_route.append(0)  # Retour au dépôt
            routes.append(current_route)
            remaining_capacity[truck] -= current_load
           
    return routes

def two_opt_optimized(route, data):
    """Optimisation 2-opt avec gestion de mémoire améliorée"""
    best = route.copy()
    improvement = True
   
    while improvement:
        improvement = False
        best_distance = calculate_route_distance(best, data)
       
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                new_distance = calculate_route_distance(new_route, data)
               
                if new_distance < best_distance:
                    best = new_route
                    best_distance = new_distance
                    improvement = True
                    break
            if improvement:
                break
               
    return best

def calculate_route_distance(route, data):
    """Calcul de distance avec pré-calcul des coordonnées"""
    return sum(math.hypot(data[route[i]]['x'] - data[route[i+1]]['x'],
                         data[route[i]]['y'] - data[route[i+1]]['y'])
               for i in range(len(route)-1))

def print_route_details(route, data):
    """Affichage détaillé des segments de route"""
    print(f"Route: {route}")
    for i in range(len(route)-1):
        x1, y1 = data[route[i]]['x'], data[route[i]]['y']
        x2, y2 = data[route[i+1]]['x'], data[route[i+1]]['y']
        dist = math.hypot(x2 - x1, y2 - y1)
        print(f"  {route[i]} ({x1},{y1}) → {route[i+1]} ({x2},{y2}): {dist:.1f} km")

def visualisation_avancee(data, routes):
    """Visualisation améliorée avec statistiques"""
    plt.figure(figsize=(14, 8))
    colors = ['#1f77b4', '#ff7f0e', '#2ca02c']
    markers = ['o', 's', 'D']
   
    # Tracé des points
    for n in data:
        node = data[n]
        plt.scatter(node['x'], node['y'],
                    s=100 + node['q']/10,
                    c='red' if n == 0 else ('gray' if node['active'] == 0 else colors[0]),
                    marker='*' if n == 0 else 'o',
                    edgecolors='black')
   
    # Tracé des routes et calcul des stats
    total_distance = 0
    for i, route in enumerate(routes):
        if len(route) < 2:
            continue
           
        x = [data[n]['x'] for n in route]
        y = [data[n]['y'] for n in route]
        distance = calculate_route_distance(route, data)
        total_distance += distance
        charge = sum(data[n]['q'] for n in route[1:-1])
       
        plt.plot(x, y, linestyle='--', marker=markers[i],
                 color=colors[i], linewidth=2,
                 label=f'Camion {i+1}: {distance:.1f} km | {charge} kg')
       
        print(f"\nCamion {i+1}:")
        print_route_details(route, data)
        print(f"Total Distance: {distance:.1f} km | Load: {charge}")
   
    plt.title("Optimisation des Tournées (k-NN Glouton + 2-opt)")
    plt.xlabel("Coordonnée X (km)")
    plt.ylabel("Coordonnée Y (km)")
    plt.grid(alpha=0.3)
    plt.legend(bbox_to_anchor=(1.05, 1), loc='upper left')
    print(f"\nDistance totale parcourue par les {N_CAMIONS} camions: {total_distance:.1f} km")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    file_path = 'D:/Doc_JAWADI/Desktop/1AGI/sem2/UE04/semaine systèmes/code/data_sujet_4.xlsx'
    try:
        # Charger les données depuis Excel au lieu de données codées en dur
        data = load_data_from_excel(file_path)
        
        # Générer et optimiser les routes
        initial_routes = greedy_knn_vrp(data, CAPACITE, K_NEIGHBORS)
        optimized_routes = [two_opt_optimized(r, data) for r in initial_routes]
        
        # Nettoyer les routes vides
        optimized_routes = [r for r in optimized_routes if len(r) > 2]
        
        # Visualiser et afficher les résultats
        visualisation_avancee(data, optimized_routes)
        
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
