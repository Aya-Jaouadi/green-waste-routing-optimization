import math
import matplotlib.pyplot as plt
import pandas as pd

def load_data_from_excel(file_path):
    try:
        df = pd.read_excel(file_path)
        print("Column names in Excel file:", list(df.columns))
        print("Data:\n", df)
        
        original_active = {
            1: 1, 2: 1, 3: 0, 4: 1, 5: 1, 6: 0, 7: 1, 8: 0, 9: 1, 10: 1,
            11: 0, 12: 0, 13: 1, 14: 0, 15: 0, 16: 0, 17: 1, 18: 0, 19: 0, 20: 1
        }
        
        data = {0: {"x": 0, "y": 0, "q": 0, "active": 0}}  # Depot
        df_transposed = df.drop('ID', axis=1).T
        
        for col in df_transposed.index:
            location_data = df_transposed.loc[col]
            data[int(col)] = {
                'x': float(location_data[0]),  # No scaling
                'y': float(location_data[1]),  # No scaling
                'q': float(location_data[2]),
                'active': original_active[int(col)]
            }
        
        print("\nLoaded data from Excel (unscaled coordinates):")
        for key, value in data.items():
            print(f"ID {key}: {value}")
        
        return data
    
    except Exception as e:
        print(f"Error loading Excel file: {str(e)}")
        raise

N_CAMIONS = 3
CAPACITE = 8000

def savings_algorithm(data, capacity):
    active_bins = [k for k in data if data[k]['active'] == 1 and k != 0]
    routes = [[0, i, 0] for i in active_bins]
    savings = []
    for i in active_bins:
        for j in active_bins:
            if i != j:
                dij = math.hypot(data[i]['x']-data[j]['x'], data[i]['y']-data[j]['y'])
                s = math.hypot(data[0]['x']-data[i]['y']) + math.hypot(data[0]['x']-data[j]['y']) - dij
                savings.append((s, i, j))
    savings.sort(reverse=True, key=lambda x: x[0])
    for s, i, j in savings:
        route_i = next((r for r in routes if i in r), None)
        route_j = next((r for r in routes if j in r), None)
        if route_i and route_j and route_i != route_j:
            total_load = sum(data[n]['q'] for n in route_i if n != 0) + sum(data[n]['q'] for n in route_j if n != 0)
            if total_load <= capacity:
                new_route = route_i[:-1] + route_j[1:]
                routes.remove(route_i)
                routes.remove(route_j)
                routes.append(new_route)
    return routes[:N_CAMIONS]

def two_opt(route, data):
    best = route
    improved = True
    while improved:
        improved = False
        for i in range(1, len(route)-2):
            for j in range(i+1, len(route)-1):
                new_route = route[:i] + route[i:j+1][::-1] + route[j+1:]
                if calculate_distance(new_route, data) < calculate_distance(best, data):
                    best = new_route
                    improved = True
        route = best
    return best

def calculate_distance(route, data):
    total = 0
    for i in range(len(route)-1):
        x1, y1 = data[route[i]]['x'], data[route[i]]['y']
        x2, y2 = data[route[i+1]]['x'], data[route[i+1]]['y']
        dist = math.hypot(x2 - x1, y2 - y1)
        total += dist
    return total

def print_route_details(route, data):
    print(f"Route: {route}")
    for i in range(len(route)-1):
        x1, y1 = data[route[i]]['x'], data[route[i]]['y']
        x2, y2 = data[route[i+1]]['x'], data[route[i+1]]['y']
        dist = math.hypot(x2 - x1, y2 - y1)
        print(f"  {route[i]} ({x1},{y1}) → {route[i+1]} ({x2},{y2}): {dist:.1f} km")

def plot_routes(data, routes):
    plt.figure(figsize=(14, 10))
    colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
    for bid in data:
        bin = data[bid]
        if bid == 0:
            plt.scatter(bin['x'], bin['y'], s=300, marker='H', c='#2C3E50', label='Dépôt')
        else:
            plt.scatter(bin['x'], bin['y'], s=50 + bin['q']/20, c='#E0E0E0' if bin['active'] == 0 else colors[0],
                        marker='D' if bin['active'] == 1 else 'o', edgecolors='black')
    for i, route in enumerate(routes):
        x_coords = [data[n]['x'] for n in route]
        y_coords = [data[n]['y'] for n in route]
        plt.plot(x_coords, y_coords, color=colors[i], linewidth=2, label=f'Camion {i+1}')
    plt.title("Optimisation des tournées - Green Waste")
    plt.legend()
    plt.show()

if __name__ == "__main__":
    file_path = 'D:/Doc_JAWADI/Desktop/1AGI/sem2/UE04/semaine systèmes/code/data_sujet_4.xlsx'
    try:
        data = load_data_from_excel(file_path)
        routes = savings_algorithm(data, CAPACITE)
        optimized_routes = [two_opt(r, data) for r in routes]
        
        print("\n=== RÉSULTATS ===")
        total_distance = 0
        for i, route in enumerate(optimized_routes):
            route_distance = calculate_distance(route, data)
            total_distance += route_distance
            total_load = sum(data[n]['q'] for n in route if n != 0)
            print(f"\nCamion {i+1}:")
            print_route_details(route, data)
            print(f"Total Distance: {route_distance:.1f} km | Load: {total_load}")
        
        print(f"\nDistance totale parcourue par les {N_CAMIONS} camions: {total_distance:.1f} km")
        
        plot_routes(data, optimized_routes)
    except Exception as e:
        print(f"Error in main execution: {str(e)}")
        