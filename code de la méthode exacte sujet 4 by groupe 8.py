# Exact optimization method for Vehicle Routing Problem (VRP)
from pulp import *
import pandas as pd
import matplotlib.pyplot as plt
import tkinter as tk
from tkinter import filedialog, messagebox, scrolledtext, ttk
from matplotlib.backends.backend_tkagg import FigureCanvasTkAgg

# Couleurs et styles
BG_COLOR = "#f0f0f0"  # Couleur de fond
BUTTON_COLOR = "#4CAF50"  # Vert
BUTTON_TEXT_COLOR = "white"  # Texte des boutons en blanc
FONT = ("Helvetica", 12)  # Police des boutons
TITLE_FONT = ("Helvetica", 16, "bold")  # Police du titre

# Variables globales
df = None
data = None
entries_capacites = []  # Pour stocker les champs de saisie des capacités

# Fonction pour lire le fichier Excel et extraire les données
def load_file():
    file_path = filedialog.askopenfilename(filetypes=[("Excel files", "*.xlsx")])
    if file_path:
        try:
            global df, data
            df = pd.read_excel(file_path, header=None)
            log_text.insert(tk.END, f"Fichier chargé : {file_path}\n", "success")
            log_text.insert(tk.END, f"Forme du DataFrame (lignes, colonnes) : {df.shape}\n", "info")

            # Extraction des données
            data = {
                "ID": [0] + df.iloc[0, 1:21].tolist(),
                "X": [0] + df.iloc[1, 1:21].tolist(),
                "Y": [0] + df.iloc[2, 1:21].tolist(),
                "Quantity": [0] + df.iloc[3, 1:21].tolist(),
                "Capacity": [0] + df.iloc[4, 1:21].tolist(),
                "FillRate": [0] + df.iloc[5, 1:21].tolist()
            }
            log_text.insert(tk.END, "Données extraites avec succès.\n", "success")
        except Exception as e:
            messagebox.showerror("Erreur", f"Erreur lors de la lecture du fichier : {e}")

# Fonction pour générer les champs de saisie des capacités
def creer_champs_capacites():
    global entries_capacites
    entries_capacites = []  # Réinitialiser la liste des champs

    # Supprimer les champs existants
    for widget in frame_capacites.winfo_children():
        widget.destroy()

    try:
        nb_camions = int(entry_nb_camions.get())
        if nb_camions <= 0:
            messagebox.showerror("Erreur", "Le nombre de camions doit être supérieur à zéro.")
            return

        # Créer des champs de saisie pour chaque camion
        for i in range(nb_camions):
            label = tk.Label(frame_capacites, text=f"Capacité du camion {i+1} (kg) :", bg=BG_COLOR, font=FONT)
            label.pack(pady=5)
            entry = tk.Entry(frame_capacites, font=FONT)
            entry.pack(pady=5)
            entries_capacites.append(entry)
    except ValueError:
        messagebox.showerror("Erreur", "Veuillez entrer un nombre valide pour le nombre de camions.")

# Fonction pour résoudre le problème et afficher les résultats
def solve_problem():
    if 'data' not in globals():
        messagebox.showwarning("Avertissement", "Veuillez d'abord charger un fichier Excel.")
        return

    if not entries_capacites:
        messagebox.showwarning("Avertissement", "Veuillez d'abord entrer les capacités des camions.")
        return

    try:
        # Récupérer les capacités des camions
        Q = {}
        for i, entry in enumerate(entries_capacites):
            capacite = int(entry.get())
            if capacite <= 0:
                messagebox.showerror("Erreur", "La capacité doit être supérieure à zéro.")
                return
            Q[i + 1] = capacite  # Les camions sont numérotés à partir de 1

        # 2. Sets and Parameters
        I = data["ID"]
        n = len(data["ID"]) - 1
        K = list(Q.keys())  # Liste des camions
        q = {i: data["Quantity"][idx] for idx, i in enumerate(data["ID"])}
        FillRate = {i: data["FillRate"][idx] for idx, i in enumerate(data["ID"])}
        depot = 0

        # Calculate distances (Euclidean)
        D = {}
        for i in I:
            for j in I:
                if i != j:
                    xi = data["X"][i]
                    yi = data["Y"][i]
                    xj = data["X"][j]
                    yj = data["Y"][j]
                    D[(i, j)] = ((xi - xj)**2 + (yi - yj)**2)**0.5
                else:
                    D[(i, j)] = 0

        # 3. Problem Definition
        prob = LpProblem("Green_Waste_Collection_With_MTZ", LpMinimize)

        # 4. Decision Variables
        X = LpVariable.dicts("X", [(i, j, k) for i in I for j in I for k in K], 0, 1, LpBinary)
        Y = LpVariable.dicts("Y", [(i, k) for i in I for k in K], 0, 1, LpBinary)
        W = LpVariable.dicts("W", [(i, k) for i in I for k in K], 0, None, LpContinuous)
        S = LpVariable.dicts("S", [i for i in I], 0, 1, LpBinary)
        U = LpVariable.dicts("U", [(i, k) for i in I for k in K], 1, n, LpInteger)

        # 5. Objective Function
        prob += lpSum(D[(i, j)] * X[(i, j, k)] for i in I for j in I for k in K), "Total_Distance"

        # 6. Constraints
        M = 100
        for i in data["ID"]:
            if i != depot:
                prob += (FillRate[i] - M * S[i] <= 50, f"Signal_Upper_{i}")
                prob += (FillRate[i] + M * (1 - S[i]) >= 50, f"Signal_Lower_{i}")

        prob += (S[depot] == 0, "Depot_No_Signal")

        for i in data["ID"]:
            if i != depot:
                prob += (lpSum(Y[(i, k)] for k in K) - S[i] == 0, f"Visit_Signaled_{i}")
        prob += (lpSum(Y[(depot, k)] for k in K) == 0, "Depot_Not_Visited")

        for k in K:
            for i in data["ID"]:
                if i != depot:
                    prob += (lpSum(X[(j, i, k)] for j in I if j != i) - Y[(i, k)] == 0, f"Flow_In_{i}_{k}")
                    prob += (lpSum(X[(i, j, k)] for j in I if j != i) - Y[(i, k)] == 0, f"Flow_Out_{i}_{k}")

        for k in K:
            prob += (lpSum(X[(depot, j, k)] for j in I if j != depot) == 1, f"Depot_Departure_{k}")
            prob += (lpSum(X[(i, depot, k)] for i in I if i != depot) == 1, f"Depot_Return_{k}")

        for k in K:
            for i in data["ID"]:
                if i != depot:
                    prob += (W[(i, k)] <= Q[k], f"Capacity_Upper_{i}_{k}")
                    prob += (W[(i, k)] >= q[i] * Y[(i, k)], f"Capacity_Lower_{i}_{k}")

        M_load = 20000
        for k in K:
            for i in data["ID"]:
                if i != depot:
                    for j in I:
                        if j != i and j != depot:
                            prob += (
                                W[(i, k)] - (W[(j, k)] + q[i] * X[(j, i, k)]) >= -M_load * (1 - X[(j, i, k)]),
                                f"Cumulative_Load_{i}_{j}_{k}"
                            )

        for k in K:
            prob += (W[(depot, k)] == 0, f"Initial_Load_{k}")

        for k in K:
            for i in I:
                for j in I:
                    if i != j and i != depot and j != depot:
                        prob += (U[(i, k)] - U[(j, k)] + n * X[(i, j, k)] <= n - 1, f"MTZ_{i}_{j}_{k}")

        # 7. Solve
        prob.solve(PULP_CBC_CMD(msg=1, timeLimit=300))

        # 8. Print Results
        log_text.insert(tk.END, f"Status: {LpStatus[prob.status]}\n", "info")
        log_text.insert(tk.END, f"Total Distance: {value(prob.objective)}\n", "info")

        log_text.insert(tk.END, "\nSignals (S_i):\n", "header")
        for i in I:
            if i != depot:
                log_text.insert(tk.END, f"Bin {i}: FillRate = {FillRate[i]}%, S_i = {value(S[i])}\n", "info")

        # Collect routes for plotting
        routes = {k: [] for k in K}
        for k in K:
            log_text.insert(tk.END, f"\nRoute for Truck {k}:\n", "header")
            total_load = 0
            total_distance = 0
            route = []
            current_bin = depot
            visited = set()
           
            while True:
                next_bin = None
                for j in I:
                    if value(X[(current_bin, j, k)]) == 1 and j not in visited:
                        next_bin = j
                        break
               
                if next_bin is None:
                    break
               
                distance_segment = D[(current_bin, next_bin)]
                total_distance += distance_segment
               
                if next_bin != depot:
                    total_load += q[next_bin]
                route.append((current_bin, next_bin, total_load, distance_segment))
               
                visited.add(next_bin)
                current_bin = next_bin
           
            if route:
                for i, j, load, dist in route:
                    log_text.insert(tk.END, f"From {i} ({data['X'][i]},{data['Y'][i]}) to {j} ({data['X'][j]},{data['Y'][j]}), "
                                  f"Load: {load}, Distance: {dist:.2f}\n", "info")
                    routes[k].append((i, j))
                log_text.insert(tk.END, f"Total Load for Truck {k}: {total_load}\n", "info")
                log_text.insert(tk.END, f"Total Distance for Truck {k}: {total_distance:.2f}\n", "info")
            else:
                log_text.insert(tk.END, "No route assigned to this truck.\n", "warning")
                routes[k] = []

        # 9. Plotting the routes
        fig, ax = plt.subplots(figsize=(8, 6))  # Taille du graphique réduite

        # Plot nodes based on signal value
        for i in I:
            if i == depot:
                ax.scatter(data["X"][i], data["Y"][i], c='red', label='Depot', s=100, marker='s')
            elif value(S[i]) == 1:
                ax.scatter(data["X"][i], data["Y"][i], c='green', label='Bins visités (signal = 1)', s=50)
            else:
                ax.scatter(data["X"][i], data["Y"][i], c='gray', label='Bins non visités (signal = 0)', s=50)
            ax.text(data["X"][i] + 0.2, data["Y"][i] + 0.2, f"{i}", fontsize=9)

        # Define colors for each truck
        colors = {1: 'blue', 2: 'purple', 3: 'orange'}

        # Plot routes for each truck
        for k in K:
            if routes[k]:
                for i, j in routes[k]:
                    ax.plot([data["X"][i], data["X"][j]], [data["Y"][i], data["Y"][j]],
                             c=colors[k], lw=2, label=f"Truck {k}" if (i, j) == routes[k][0] else "")

        # Add legend and labels
        handles, labels = ax.get_legend_handles_labels()
        by_label = dict(zip(labels, handles))  # Remove duplicates
        ax.legend(by_label.values(), by_label.keys())
        ax.set_title("Routes des camions pour la collecte des déchets verts")
        ax.set_xlabel("X Coordinate")
        ax.set_ylabel("Y Coordinate")
        ax.grid(True)

        # Embed the plot in the Tkinter window
        if hasattr(solve_problem, 'canvas'):
            solve_problem.canvas.get_tk_widget().destroy()  # Supprimer l'ancien graphique

        solve_problem.canvas = FigureCanvasTkAgg(fig, master=plot_frame)
        solve_problem.canvas.draw()
        solve_problem.canvas.get_tk_widget().pack(side=tk.LEFT, fill=tk.BOTH, expand=True)

    except Exception as e:
        messagebox.showerror("Erreur", f"Une erreur s'est produite : {e}")

# Interface graphique
root = tk.Tk()
root.title("optimisation de la collecte des déchets du start up “Green Waste”")
root.geometry("1200x700")  # Taille de la fenêtre réduite
root.config(bg=BG_COLOR)

# Titre
label_title = tk.Label(root, text="Optimisation de la collecte du start up “Green Waste”", font=TITLE_FONT, bg=BG_COLOR)
label_title.grid(row=0, column=0, columnspan=2, pady=10)

# Frame pour le nombre de camions
frame_nb_camions = tk.Frame(root, bg=BG_COLOR)
frame_nb_camions.grid(row=1, column=0, columnspan=2, pady=10)

label_nb_camions = tk.Label(frame_nb_camions, text="Nombre de camions :", bg=BG_COLOR, font=FONT)
label_nb_camions.pack(side=tk.LEFT, padx=10)

entry_nb_camions = tk.Entry(frame_nb_camions, font=FONT)
entry_nb_camions.pack(side=tk.LEFT, padx=10)

button_valider_camions = tk.Button(frame_nb_camions, text="Valider", command=creer_champs_capacites, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR, font=FONT)
button_valider_camions.pack(side=tk.LEFT, padx=10)

# Frame pour les capacités des camions
frame_capacites = tk.Frame(root, bg=BG_COLOR)
frame_capacites.grid(row=2, column=0, columnspan=2, pady=10)

# Frame pour les boutons
button_frame = tk.Frame(root, bg=BG_COLOR)
button_frame.grid(row=3, column=0, columnspan=2, pady=10)

# Bouton pour charger un fichier
load_button = tk.Button(button_frame, text="Charger un fichier Excel", command=load_file, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR, font=FONT)
load_button.pack(side=tk.LEFT, padx=10)

# Bouton pour résoudre le problème
solve_button = tk.Button(button_frame, text="Résoudre le problème", command=solve_problem, bg=BUTTON_COLOR, fg=BUTTON_TEXT_COLOR, font=FONT)
solve_button.pack(side=tk.LEFT, padx=10)

# Frame pour le texte de log
log_frame = tk.Frame(root, bg=BG_COLOR)
log_frame.grid(row=4, column=0, sticky="nsew", padx=10, pady=10)

# Zone de texte pour les logs
log_text = scrolledtext.ScrolledText(log_frame, wrap=tk.WORD, width=60, height=20, font=FONT, bg="white", fg="black")
log_text.pack(fill=tk.BOTH, expand=True)

# Ajouter des tags pour le style du texte
log_text.tag_configure("success", foreground="green")
log_text.tag_configure("info", foreground="black")
log_text.tag_configure("warning", foreground="orange")
log_text.tag_configure("header", font=("Helvetica", 12, "bold"))

# Frame pour le graphique
plot_frame = tk.Frame(root, bg=BG_COLOR)
plot_frame.grid(row=4, column=1, sticky="nsew", padx=10, pady=10)

# Configurer la grille pour redimensionner les colonnes
root.grid_columnconfigure(0, weight=1)
root.grid_columnconfigure(1, weight=1)
root.grid_rowconfigure(4, weight=1)

# Lancer l'interface
root.mainloop()
